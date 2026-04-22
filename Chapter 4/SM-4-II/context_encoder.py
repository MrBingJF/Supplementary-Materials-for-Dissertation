"""
上下文感知元学习模块 - 变分自编码器(VAE)上下文编码器
核心功能: 从历史轨迹推断隐变量z，编码全局电力供需意图，缓解多智能体环境非平稳性。
"""

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

EPS = 1e-8


def mlp(x, hidden_sizes, activation=tf.nn.relu, output_activation=None, name='mlp'):
    with tf.variable_scope(name):
        for i, h in enumerate(hidden_sizes[:-1]):
            x = tf.layers.dense(x, units=h, activation=activation, name=f'fc_{i}')
        x = tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, name='fc_out')
    return x


class ContextEncoder:
    """从历史轨迹中推断环境隐变量z(捕获状态转移、交互行为及不确定性特征)"""
    
    def __init__(self, 
                 context_input_dim,      
                 latent_dim=16,          
                 encoder_hidden=[128, 64],  
                 decoder_hidden=[64, 128],  
                 aggregator_hidden=[64, 32], 
                 name='context_encoder'):
        self.context_input_dim = context_input_dim
        self.latent_dim = latent_dim
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.aggregator_hidden = aggregator_hidden
        self.name = name
        
    def build_encoder(self, context_input, reuse=False):
        with tf.variable_scope(self.name + '/encoder', reuse=reuse):
            batch_size = tf.shape(context_input)[0]
            seq_len = tf.shape(context_input)[1]
            
            flat_input = tf.reshape(context_input, [-1, self.context_input_dim])
            
            h = flat_input
            for i, hidden_size in enumerate(self.encoder_hidden):
                h = tf.layers.dense(h, hidden_size, activation=tf.nn.relu, name=f'enc_fc_{i}')
            
            step_embedding = tf.layers.dense(h, self.latent_dim * 2, name='step_embed')
            step_embedding = tf.reshape(step_embedding, [batch_size, seq_len, self.latent_dim * 2])
            
            attention_weights = self._attention_aggregator(step_embedding, reuse=reuse)
            aggregated = tf.reduce_sum(step_embedding * attention_weights, axis=1)  
            
            z_mu = aggregated[:, :self.latent_dim]
            z_log_var = aggregated[:, self.latent_dim:]
            z_log_var = tf.clip_by_value(z_log_var, -10, 2)  
            
            z = self._reparameterize(z_mu, z_log_var)
            
            return z_mu, z_log_var, z
    
    def _attention_aggregator(self, embeddings, reuse=False):
        with tf.variable_scope(self.name + '/attention', reuse=reuse):
            h = embeddings
            for i, hidden_size in enumerate(self.aggregator_hidden):
                h = tf.layers.dense(h, hidden_size, activation=tf.nn.tanh, name=f'att_fc_{i}')
            
            scores = tf.layers.dense(h, 1, name='att_score') 
            attention_weights = tf.nn.softmax(scores, axis=1)
            
            return attention_weights
    
    def _reparameterize(self, mu, log_var):
        """重参数化技巧: z = mu + sigma * epsilon, epsilon ~ N(0, 1)"""
        std = tf.exp(0.5 * log_var)
        eps = tf.random_normal(tf.shape(mu))
        return mu + std * eps
    
    def build_decoder(self, z, target_dim, reuse=False):
        with tf.variable_scope(self.name + '/decoder', reuse=reuse):
            h = z
            for i, hidden_size in enumerate(self.decoder_hidden):
                h = tf.layers.dense(h, hidden_size, activation=tf.nn.relu, name=f'dec_fc_{i}')
            
            reconstruction = tf.layers.dense(h, target_dim, name='dec_out')
            return reconstruction
    
    def compute_kl_loss(self, z_mu, z_log_var):
        """KL(q(z|c) || p(z)) where p(z) = N(0, I)"""
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mu) - tf.exp(z_log_var), 
            axis=-1
        )
        return tf.reduce_mean(kl_loss)
    
    def compute_reconstruction_loss(self, target, reconstruction):
        recon_loss = tf.reduce_mean(tf.square(target - reconstruction))
        return recon_loss


class MultiAgentContextEncoder:
    """为每个智能体维护独立上下文投影，同时共享全局上下文以感知供需平衡"""
    
    def __init__(self,
                 obs_dim1, obs_dim2,          
                 act_dim1, act_dim2,          
                 global_state_dim,             
                 latent_dim=16,                
                 encoder_hidden=[128, 64],     
                 decoder_hidden=[64, 128],     
                 context_len=5):               
        self.obs_dim1 = obs_dim1
        self.obs_dim2 = obs_dim2
        self.act_dim1 = act_dim1
        self.act_dim2 = act_dim2
        self.global_state_dim = global_state_dim
        self.latent_dim = latent_dim
        self.context_len = context_len
        
        self.context_input_dim = global_state_dim + act_dim1 + act_dim2 + 2 + global_state_dim
        
        self.shared_encoder = ContextEncoder(
            context_input_dim=self.context_input_dim,
            latent_dim=latent_dim,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            name='shared_context'
        )
        
        self.agent1_latent_dim = latent_dim
        self.agent2_latent_dim = latent_dim
        
    def build_context_inference(self, context_batch, reuse=False):
        z_mu, z_log_var, z = self.shared_encoder.build_encoder(context_batch, reuse=reuse)
        
        with tf.variable_scope('agent_context_proj', reuse=reuse):
            z1 = tf.layers.dense(z, self.agent1_latent_dim, activation=tf.nn.tanh, name='proj_agent1')
            z2 = tf.layers.dense(z, self.agent2_latent_dim, activation=tf.nn.tanh, name='proj_agent2')
        
        return z_mu, z_log_var, z, z1, z2
    
    def build_decoder(self, z, reuse=False):
        pred_reward = self.shared_encoder.build_decoder(z, target_dim=2, reuse=reuse)
        return pred_reward


class LocalContextBuffer:
    """
    局部上下文缓冲区 (遵循CTDE原则)
    仅存储自身可观测信息 (o_i, a_i, r_i, c_i) 确保分布式执行不越权获取全局信息。
    """
    
    def __init__(self, context_len, obs_dim, act_dim):
        self.context_len = context_len
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_dim = obs_dim + act_dim + 2  
        
        self._history = []
        
    def add_transition(self, obs, action, reward, cost):
        transition = np.concatenate([
            obs, action, [reward, cost]
        ])
        self._history.append(transition)
        
        if len(self._history) > self.context_len * 2:
            self._history = self._history[-self.context_len:]
    
    def get_context_for_decision(self):
        """获取用于决策的上下文(t-H 到 t-1)，避免因果穿越(预知a_t/r_t)"""
        if len(self._history) >= self.context_len:
            return np.array(self._history[-self.context_len:])
        elif len(self._history) > 0:
            context = np.array(self._history)
            padding = np.zeros((self.context_len - len(context), self.context_dim))
            return np.vstack([padding, context])
        else:
            return np.zeros((self.context_len, self.context_dim))
    
    def reset(self):
        self._history = []


class ContextBuffer:
    """[已废弃] 全局上下文缓冲区 (兼容旧代码保留，建议使用 LocalContextBuffer)"""
    
    def __init__(self, context_len, context_dim, batch_size=256):
        self.context_len = context_len
        self.context_dim = context_dim
        self.batch_size = batch_size
        self.current_context = []
        self.episode_contexts = []
        self.max_episodes = 1000
        
    def add_transition(self, global_state, action1, action2, reward1, reward2, 
                       cost1, cost2, global_state_next):
        transition = np.concatenate([
            global_state, action1, action2, 
            [reward1, reward2],
            [cost1, cost2],
            global_state_next
        ])
        self.current_context.append(transition)
        
    def end_episode(self):
        if len(self.current_context) >= self.context_len:
            self.episode_contexts.append(np.array(self.current_context))
            if len(self.episode_contexts) > self.max_episodes:
                self.episode_contexts.pop(0)
        self.current_context = []
        
    def get_context(self):
        if len(self.current_context) >= self.context_len:
            return np.array(self.current_context[-self.context_len:])
        elif len(self.current_context) > 0:
            context = np.array(self.current_context)
            padding = np.zeros((self.context_len - len(context), self.context_dim))
            return np.vstack([padding, context])
        else:
            return np.zeros((self.context_len, self.context_dim))
    
    def sample_contexts(self, batch_size):
        if len(self.episode_contexts) == 0:
            return np.zeros((batch_size, self.context_len, self.context_dim))
        
        contexts = []
        for _ in range(batch_size):
            ep_idx = np.random.randint(len(self.episode_contexts))
            ep = self.episode_contexts[ep_idx]
            
            if len(ep) >= self.context_len:
                start_idx = np.random.randint(0, len(ep) - self.context_len + 1)
                context = ep[start_idx:start_idx + self.context_len]
            else:
                context = np.vstack([
                    np.zeros((self.context_len - len(ep), self.context_dim)),
                    ep
                ])
            contexts.append(context)
            
        return np.array(contexts)


def build_context_conditioned_actor(obs_ph, z_ph, a_ph, hidden_sizes, activation, 
                                    output_activation, name='context_pi'):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    
    act_dim = a_ph.shape.as_list()[-1]
    
    with tf.variable_scope(name):
        x = tf.concat([obs_ph, z_ph], axis=-1)
        
        for i, h in enumerate(hidden_sizes[:-1]):
            x = tf.layers.dense(x, h, activation=activation, name=f'fc_{i}')
        x = tf.layers.dense(x, hidden_sizes[-1], activation=activation, name=f'fc_{len(hidden_sizes)-1}')
        
        mu = tf.layers.dense(x, act_dim, activation=output_activation, name='mu')
        log_std = tf.layers.dense(x, act_dim, activation=None, name='log_std')
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        std = tf.exp(log_std)
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        
        pre_sum = -0.5 * (((pi - mu) / (std + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        logp_pi = tf.reduce_sum(pre_sum, axis=1)
        
        logp_pi -= tf.reduce_sum(2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)
        mu = tf.tanh(mu)
        pi = tf.tanh(pi)
        
    return mu, pi, logp_pi


def build_context_conditioned_critic(s_ph, a_ph, pi_ph, z_ph, hidden_sizes, 
                                     activation, name='context_q'):
    def build_q(x, name_suffix=''):
        with tf.variable_scope(name + name_suffix, reuse=tf.AUTO_REUSE):
            for i, h in enumerate(hidden_sizes):
                x = tf.layers.dense(x, h, activation=activation, name=f'fc_{i}')
            q = tf.layers.dense(x, 1, activation=None, name='q_out')
            return tf.squeeze(q, axis=1)
    
    x_a = tf.concat([s_ph, a_ph, z_ph], axis=-1)
    x_pi = tf.concat([s_ph, pi_ph, z_ph], axis=-1)
    
    q = build_q(x_a)
    q_pi = build_q(x_pi)
    
    return q, q_pi


if __name__ == '__main__':
    context_dim = 100
    latent_dim = 16
    seq_len = 5
    batch_size = 32
    
    context_ph = tf.placeholder(tf.float32, [None, seq_len, context_dim])
    
    encoder = ContextEncoder(
        context_input_dim=context_dim,
        latent_dim=latent_dim
    )
    
    z_mu, z_log_var, z = encoder.build_encoder(context_ph)
    
    print(f"z_mu shape: {z_mu.shape}")
    print(f"z_log_var shape: {z_log_var.shape}")
    print(f"z shape: {z.shape}")
    
    kl_loss = encoder.compute_kl_loss(z_mu, z_log_var)
    print(f"KL loss shape: {kl_loss.shape}")