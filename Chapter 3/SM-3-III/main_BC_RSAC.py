'''
Chapter3:BC-RSAC
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  
import tensorflow._api.v2.compat.v1 as tf
tf.reset_default_graph()
tf.disable_v2_behavior()
import shutil
import numpy as np
import pandas as pd
import time
import datetime
import seaborn as sns
from logx import EpochLogger
from mpi_tf import sync_all_params, MpiAdamOptimizer
from mpi_tools import mpi_fork, mpi_sum, proc_id, num_procs
from scipy.stats import norm
import globalvar as gl
import json
import argparse
from run_utils import setup_logger_kwargs
import save_data
import matplotlib.pyplot as plt
from env3 import unPower 

# 获取全局参数
n_T = gl.get_value('n_T')
n_wind = gl.get_value('n_wind')
n_gen = gl.get_value('n_gen')
n_E = gl.get_value('n_E')
n_wind_data = gl.get_value('n_wind_data')  
obs_dim = gl.get_value('state_dim')
act_dim = gl.get_value('action_dim')
P_load = gl.get_value('P_load')
P_wind = gl.get_value('P_wind')
load_capacity = gl.get_value('load_capacity')
wind_capacity = gl.get_value('wind_capacity')
pg_max = gl.get_value('pg_max')
pg_min = gl.get_value('pg_min')
ROCOFmax = gl.get_value('ROCOFmax')
delt_fmax = gl.get_value('delt_fmax')
EPS = 1e-8

snow = datetime.datetime.now()  
seed_now = int(str(snow.month)+str(snow.day)+str(snow.hour))
                
#============================================Configuration Parameters================================================
parser = argparse.ArgumentParser()
parser.add_argument('--Train', type=bool, default=True)
parser.add_argument('--Continue', type=bool, default=False)
parser.add_argument('--Portion', type=bool, default=False)
parser.add_argument('--SaveFig', type=bool, default=False)
parser.add_argument('--load_buffer', type=bool, default=False)
parser.add_argument('--Model_version', type=int, default=12000)
parser.add_argument('--epochs', type=int, default=8001) 
parser.add_argument('--steps_per_epoch', type=int, default=3)
parser.add_argument('--update_freq', type=int, default=60)
parser.add_argument('--save_freq', default=2000, type=int)

# 双成本约束参数
parser.add_argument('--cost_lim1', type=float, default=0.1)
parser.add_argument('--cost_lim2', type=float, default=0.1)
parser.add_argument('--cl', type=float, default=0.95)
                                                                                          
parser.add_argument('--hidden_sizes_actor', type=list, default=[128, 256, 256, 256])
parser.add_argument('--hidden_sizes_critic', type=list, default=[256, 256, 256, 64])
parser.add_argument('--hidden_sizes_var', type=list, default=[256, 256, 256, 64])
parser.add_argument('--cnn_actor', type=list, default=[])
parser.add_argument('--cnn_critic', type=list, default=[])
parser.add_argument('--cnn_var', type=list, default=[])

parser.add_argument('--hid', type=int, default=256)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=seed_now)
parser.add_argument('--exp_name', type=str, default='sac')   
parser.add_argument('--cpu', type=int, default=1)
parser.add_argument('--render', default=False, action='store_true')
parser.add_argument('--local_start_steps', default=3600, type=int) 
parser.add_argument('--local_update_after', default=3600, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--fixed_entropy_bonus', default=None, type=float)
parser.add_argument('--entropy_constraint', type=float, default=-1)

parser.add_argument('--fixed_cost_penalty1', default=None, type=float)
parser.add_argument('--fixed_cost_penalty2', default=None, type=float)
parser.add_argument('--cost_constraint1', type=float, default=None)
parser.add_argument('--cost_constraint2', type=float, default=None)

parser.add_argument('--lr_s', type=int, default=50)
parser.add_argument('--damp_s', type=int, default=10)
parser.add_argument('--logger_kwargs_str', type=json.loads, default='{"output_dir": "./logger"}')
args = parser.parse_args()
#==================================================================================================================

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def cnn_mlp(x, hidden_sizes=(64,), activation=tf.tanh, output_activation=None, cnn_sizes=[16, 32]):
    j = 1
    if cnn_sizes:
        dim = int(x.shape[1])
        x = tf.reshape(x, (-1, dim, 1))
        for i in cnn_sizes:
            x = tf.layers.conv1d(inputs=x, filters=i, kernel_size=3, padding='same', 
                                 use_bias=False, activation=None, name='cnn_%d'%(j))
            x = tf.layers.batch_normalization(inputs=x, training=True, name='bn_%d'%(j))
            x = tf.tanh(x)
            j += 1
        x = tf.reshape(x, (-1, dim * cnn_sizes[-1]))
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation, name='fc_%d'%(j))
        j += 1
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, name='fc_%d'%(j))

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    return sum([np.prod(var.shape.as_list()) for var in get_vars(scope)])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def get_target_update(main_name, target_name, polyak):
    """基于主网络参数软更新目标网络"""
    main_vars = {x.name: x for x in get_vars(main_name)}
    targ_vars = {x.name: x for x in get_vars(target_name)}
    assign_ops = []
    for v_targ in targ_vars:
        v_main = v_targ.replace(target_name, main_name, 1)
        assign_op = tf.assign(targ_vars[v_targ], polyak * targ_vars[v_targ] + (1 - polyak) * main_vars[v_main])
        assign_ops.append(assign_op)
    return tf.group(assign_ops)

""" Policies """
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, cnn_sizes):
    act_dim = a.shape.as_list()[-1]
    net = cnn_mlp(x, list(hidden_sizes), activation, activation, list(cnn_sizes))
    mu = tf.layers.dense(net, act_dim, activation=output_activation, name='fc_mu')
    log_std = tf.layers.dense(net, act_dim, activation=None, name='fc_std')
    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    logp_pi -= tf.reduce_sum(2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)
    return tf.tanh(mu), tf.tanh(pi), logp_pi

""" Actors and Critics """
def mlp_actor(x, a, name='pi', hidden_sizes=(64, 64), activation=tf.nn.relu,
              output_activation=None, policy=mlp_gaussian_policy, action_space=None, cnn_sizes=[8,16]):
    with tf.variable_scope(name):
        mu, pi, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, cnn_sizes)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
    return mu, pi, logp_pi

def mlp_var(x, a, pi, name, hidden_sizes=(64, 64), activation=tf.nn.relu,
            output_activation=None, action_space=None, cnn_sizes=[]):
    fn_mlp = lambda x: tf.squeeze(cnn_mlp(x=x, hidden_sizes=list(hidden_sizes) + [1],
                                      activation=activation, output_activation=None, cnn_sizes=cnn_sizes), axis=1)
    with tf.variable_scope(name):
        var = tf.nn.softplus(fn_mlp(tf.concat([x, a], axis=-1)))
    with tf.variable_scope(name, reuse=True):
        var_pi = tf.nn.softplus(fn_mlp(tf.concat([x, pi], axis=-1)))
    return var, var_pi

def mlp_critic(x, a, pi, name, hidden_sizes=(64, 64), activation=tf.nn.relu,
               output_activation=None, action_space=None, cnn_sizes=[]):
    fn_mlp = lambda x: tf.squeeze(cnn_mlp(x=x, hidden_sizes=list(hidden_sizes) + [1],
                                      activation=activation, output_activation=None, cnn_sizes=cnn_sizes), axis=1)
    with tf.variable_scope(name):
        critic = fn_mlp(tf.concat([x, a], axis=-1))
    with tf.variable_scope(name, reuse=True):
        critic_pi = fn_mlp(tf.concat([x, pi], axis=-1))
    return critic, critic_pi

def Fig(logger_Dict, item='EpCost', c=None, Title=None, SaveFig=True, tsplot=True):
    try:
        Epoch = np.concatenate((logger_Dict['Epoch'], logger_Dict['Epoch'], logger_Dict['Epoch'])) if tsplot else logger_Dict['Epoch']
        if tsplot:
            data = np.concatenate((logger_Dict['Average'+item], logger_Dict['Min'+item], logger_Dict['Max'+item]))
            sns.set(style="whitegrid")
            sns.lineplot(x=Epoch, y=data, color=c, lw=0.8)
        else:
            plt.plot(Epoch, logger_Dict[item])
            
        plt.xlabel("Epoch")
        plt.ylabel(item)
        plt.title(Title if Title else f'The Expectations of {item} of Each Epoch')
        
        if SaveFig:
            now = datetime.datetime.now()
            savefname = f'./Debug_record/{now.year}.{now.month}.{now.day}/Figture'
            os.makedirs(savefname, exist_ok=True)
            plt.savefig(f'{savefname}/{now.hour}-{now.minute}-{item}.png', dpi=500)
        plt.show()
    except KeyError:
        print(f'在logger_Dict中无法找到与“{item}”对应的数据')

class ReplayBuffer:
    """ BC-RSAC经验池 (含分布式成本支持) """
    def __init__(self, obs_dim, act_dim, size, load_buffer=True, Path_xls='./logger/Buffer.xlsx'):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.costs1_buf = np.zeros(size, dtype=np.float32)
        self.costs2_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        
        if load_buffer:
            try: 
                excel_data = pd.read_excel(Path_xls, sheet_name=None)
                obs1 = np.array(excel_data['obs1'])
                
                u = min(size, len(excel_data['rews']))
                self.obs1_buf[:u] = obs1[:u]
                self.obs2_buf[:u] = np.array(excel_data['obs2'])[:u]
                self.acts_buf[:u] = np.array(excel_data['acts'])[:u]
                self.rews_buf[:u] = np.array(excel_data['rews'])[:u].reshape(-1)
                self.costs1_buf[:u] = np.array(excel_data['costs1'])[:u].reshape(-1)
                self.costs2_buf[:u] = np.array(excel_data['costs2'])[:u].reshape(-1)
                self.done_buf[:u] = np.array(excel_data['done'])[:u].reshape(-1)
                
                zero_idx = np.where(obs1[:u, obs_dim-1] == 0)[0]
                init_ptr = zero_idx[0] if zero_idx.size > 0 else u
                
                self.ptr = init_ptr % self.max_size
                self.size = init_ptr
                print('经验池已加载\n')         
            except Exception as e:
                print('经验池加载失败，文件不存在或参数不对应:', e)

    def store(self, obs, act, rew, next_obs, done, cost1, cost2):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.costs1_buf[self.ptr] = cost1
        self.costs2_buf[self.ptr] = cost2
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs], obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs], rews=self.rews_buf[idxs],
                    costs1=self.costs1_buf[idxs], costs2=self.costs2_buf[idxs],
                    done=self.done_buf[idxs])
    
    def savebuffer(self, Path_xls):
        data = dict(obs1=self.obs1_buf, obs2=self.obs2_buf, acts=self.acts_buf,
                    rews=self.rews_buf, costs1=self.costs1_buf, costs2=self.costs2_buf, done=self.done_buf)
        with pd.ExcelWriter(Path_xls) as writer:   
            for k, v in data.items():                                 
                pd.DataFrame(v).to_excel(writer, sheet_name=k, index=False)
                
""" BC-RSAC """
def BC_RSAC(env_fn, actor_fn=mlp_actor, critic_fn=mlp_critic, var_fn=mlp_var, 
        ac_kwargs_actor=dict(), ac_kwargs_critic=dict(), ac_kwargs_var=dict(),
        seed=0, steps_per_epoch=1200, epochs=8001, replay_size=1200 * 10 * 24, gamma=0.99, cl=0.5,
        polyak=0.995, batch_size=1024, local_start_steps=600,
        max_ep_len=n_T, logger_kwargs=dict(), save_freq=50, local_update_after=int(1e3),
        update_freq=120, render=False, fixed_entropy_bonus=None, entropy_constraint=-1.0,
        fixed_cost_penalty1=None, cost_constraint1=None, cost_lim1=None,
        fixed_cost_penalty2=None, cost_constraint2=None, cost_lim2=None,
        reward_scale=1, lr_scale=1, damp_scale=0, Train=True, Continue_Training=False,
        Model_Portion=False, load_buffer=True):
    
    use_costs = (fixed_cost_penalty1 or cost_constraint1 or cost_lim1) or (fixed_cost_penalty2 or cost_constraint2 or cost_lim2)
    pdf_cdf = cl ** (-1) * norm.pdf(norm.ppf(cl))

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env, test_env = env_fn(), env_fn()
    seed += 200 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    x_ph, a_ph, x2_ph, r_ph, d_ph, c1_ph, c2_ph = placeholders(obs_dim, act_dim, obs_dim, None, None, None, None)
    
    # 动态学习率占位符 
    lr_ph = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')

    # 主网络构建
    with tf.variable_scope('main'):
        mu, pi, logp_pi = actor_fn(x_ph, a_ph, **ac_kwargs_actor)
        qr1, qr1_pi = critic_fn(x_ph, a_ph, pi, name='qr1', **ac_kwargs_critic)
        qr2, qr2_pi = critic_fn(x_ph, a_ph, pi, name='qr2', **ac_kwargs_critic)
        
        # 分布式成本评估网络 (期望+方差)
        qc1, qc1_pi = critic_fn(x_ph, a_ph, pi, name='qc1', **ac_kwargs_critic)
        qc1_var, qc1_pi_var = var_fn(x_ph, a_ph, pi, name='qc1_var', **ac_kwargs_var)
        
        qc2, qc2_pi = critic_fn(x_ph, a_ph, pi, name='qc2', **ac_kwargs_critic)
        qc2_var, qc2_pi_var = var_fn(x_ph, a_ph, pi, name='qc2_var', **ac_kwargs_var)

    with tf.variable_scope('main', reuse=True):
        _, pi2, logp_pi2 = actor_fn(x2_ph, a_ph, **ac_kwargs_actor)

    # 目标网络构建
    with tf.variable_scope('target'):
        _, qr1_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qr1', **ac_kwargs_critic)
        _, qr2_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qr2', **ac_kwargs_critic)
        _, qc1_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qc1', **ac_kwargs_critic)
        _, qc1_pi_var_targ = var_fn(x2_ph, a_ph, pi2, name='qc1_var', **ac_kwargs_var)
        _, qc2_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qc2', **ac_kwargs_critic)
        _, qc2_pi_var_targ = var_fn(x2_ph, a_ph, pi2, name='qc2_var', **ac_kwargs_var)

    # 熵约束
    if fixed_entropy_bonus is None:
        with tf.variable_scope('entreg'):
            soft_alpha = tf.get_variable('soft_alpha', initializer=0.0, trainable=True, dtype=tf.float32)
        alpha = tf.nn.softplus(soft_alpha)
    else:
        alpha = tf.constant(fixed_entropy_bonus)
    log_alpha = tf.log(tf.clip_by_value(alpha, 1e-8, 1e8))

    # 分布式成本乘子约束 (Beta)
    if use_costs:
        with tf.variable_scope('costpen'):
            if fixed_cost_penalty1 is None:
                soft_beta1 = tf.get_variable('soft_beta1', initializer=0.0, trainable=True, dtype=tf.float32)
                beta1 = tf.nn.softplus(soft_beta1)
            else:
                beta1 = tf.constant(fixed_cost_penalty1)
            log_beta1 = tf.log(tf.clip_by_value(beta1, 1e-8, 1e8))
                
            if fixed_cost_penalty2 is None:
                soft_beta2 = tf.get_variable('soft_beta2', initializer=0.0, trainable=True, dtype=tf.float32)
                beta2 = tf.nn.softplus(soft_beta2)
            else:
                beta2 = tf.constant(fixed_cost_penalty2)
            log_beta2 = tf.log(tf.clip_by_value(beta2, 1e-8, 1e8))
    else:
        beta1, beta2 = 0.0, 0.0

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, load_buffer=load_buffer)

    if proc_id() == 0:
        var_counts = tuple(count_vars(scope) for scope in ['main/pi', 'main/qr1', 'main/qr2', 'main/qc1', 'main/qc1_var', 'main/qc2', 'main/qc2_var', 'main'])
        print('\nParams: pi:%d, qr1:%d, qr2:%d, qc1:%d, qc1_v:%d, qc2:%d, qc2_v:%d, total:%d\n' % var_counts)

    min_q_pi = tf.minimum(qr1_pi, qr2_pi)
    min_q_pi_targ = tf.minimum(qr1_pi_targ, qr2_pi_targ)

    qc1_var = tf.clip_by_value(qc1_var, 1e-8, 1e8)
    qc1_pi_var = tf.clip_by_value(qc1_pi_var, 1e-8, 1e8)
    qc1_pi_var_targ = tf.clip_by_value(qc1_pi_var_targ, 1e-8, 1e8)

    qc2_var = tf.clip_by_value(qc2_var, 1e-8, 1e8)
    qc2_pi_var = tf.clip_by_value(qc2_pi_var, 1e-8, 1e8)
    qc2_pi_var_targ = tf.clip_by_value(qc2_pi_var_targ, 1e-8, 1e8)

    # 计算贝尔曼目标
    q_backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * (min_q_pi_targ - alpha * logp_pi2))
    
    qc1_backup = tf.stop_gradient(c1_ph + gamma * (1 - d_ph) * qc1_pi_targ)
    qc1_var_backup = tf.stop_gradient(c1_ph**2 + 2*gamma*c1_ph*qc1_pi_targ + gamma**2*qc1_pi_var_targ + gamma**2*qc1_pi_targ**2 - qc1**2)
    qc1_var_backup = tf.clip_by_value(qc1_var_backup, 1e-8, 1e8)
    
    qc2_backup = tf.stop_gradient(c2_ph + gamma * (1 - d_ph) * qc2_pi_targ)
    qc2_var_backup = tf.stop_gradient(c2_ph**2 + 2*gamma*c2_ph*qc2_pi_targ + gamma**2*qc2_pi_var_targ + gamma**2*qc2_pi_targ**2 - qc2**2)
    qc2_var_backup = tf.clip_by_value(qc2_var_backup, 1e-8, 1e8)

    cost_constraint1_val = cost_lim1 * (1 - gamma ** max_ep_len) / (1 - gamma) / max_ep_len
    cost_constraint2_val = cost_lim2 * (1 - gamma ** max_ep_len) / (1 - gamma) / max_ep_len
    
    damp1 = damp_scale * tf.reduce_mean(cost_constraint1_val - qc1 - pdf_cdf * tf.sqrt(qc1_var))
    damp2 = damp_scale * tf.reduce_mean(cost_constraint2_val - qc2 - pdf_cdf * tf.sqrt(qc2_var))

    # 损失函数定义
    pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi 
                             + (beta1 - damp1) * (qc1_pi + pdf_cdf * (qc1_pi_var ** 0.5))
                             + (beta2 - damp2) * (qc2_pi + pdf_cdf * (qc2_pi_var ** 0.5)))
                             
    qr1_loss = 0.5 * tf.reduce_mean((q_backup - qr1) ** 2)
    qr2_loss = 0.5 * tf.reduce_mean((q_backup - qr2) ** 2)
    
    qc1_loss = 0.5 * tf.reduce_mean((qc1_backup - qc1) ** 2)
    qc1_var_loss = 0.5 * tf.reduce_mean(qc1_var + qc1_var_backup - 2 * ((qc1_var * qc1_var_backup) ** 0.5))
    
    qc2_loss = 0.5 * tf.reduce_mean((qc2_backup - qc2) ** 2)
    qc2_var_loss = 0.5 * tf.reduce_mean(qc2_var + qc2_var_backup - 2 * ((qc2_var * qc2_var_backup) ** 0.5))
    
    q_loss = qr1_loss + qr2_loss + qc1_loss + qc1_var_loss + qc2_loss + qc2_var_loss

    entropy_constraint *= act_dim
    pi_entropy = -tf.reduce_mean(logp_pi)
    alpha_loss = - alpha * (entropy_constraint - pi_entropy)

    if use_costs:
        cost_constraint1 = cost_constraint1 if cost_constraint1 is None else cost_constraint1_val
        cost_constraint2 = cost_constraint2 if cost_constraint2 is None else cost_constraint2_val
        beta1_loss = beta1 * (cost_constraint1 - qc1 - pdf_cdf * tf.sqrt(qc1_var))
        beta2_loss = beta2 * (cost_constraint2 - qc2 - pdf_cdf * tf.sqrt(qc2_var))
        beta_loss = beta1_loss + beta2_loss

    # 优化器 - 接入动态学习率 lr_ph
    train_pi_op = MpiAdamOptimizer(learning_rate=lr_ph).minimize(pi_loss, var_list=get_vars('main/pi'), name='train_pi')
    with tf.control_dependencies([train_pi_op]):
        train_q_op = MpiAdamOptimizer(learning_rate=lr_ph).minimize(q_loss, var_list=get_vars('main/q'), name='train_q')
       
    if fixed_entropy_bonus is None:
        entreg_optimizer = MpiAdamOptimizer(learning_rate=lr_ph)
        with tf.control_dependencies([train_q_op]):
            train_entreg_op = entreg_optimizer.minimize(alpha_loss, var_list=get_vars('entreg'))

    if use_costs and (fixed_cost_penalty1 is None or fixed_cost_penalty2 is None):
        costpen_optimizer = MpiAdamOptimizer(learning_rate=lr_ph * lr_scale)
        deps = [train_entreg_op] if fixed_entropy_bonus is None else [train_q_op]
        with tf.control_dependencies(deps):
            train_costpen_op = costpen_optimizer.minimize(beta_loss, var_list=get_vars('costpen'))

    target_update = get_target_update('main', 'target', polyak)
    with tf.control_dependencies([train_pi_op, train_q_op]):
        grouped_update = tf.group([target_update])

    if fixed_entropy_bonus is None:
        grouped_update = tf.group([grouped_update, train_entreg_op])
    if use_costs and (fixed_cost_penalty1 is None or fixed_cost_penalty2 is None):
        grouped_update = tf.group([grouped_update, train_costpen_op])
        
    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1, -1)})[0]

    def test_agent(num_wind=0, deterministic=True, n_test_sce=100):
        o = test_env.reset(n=num_wind, Train=Train)
        ep_ret, ep_cost1, ep_cost2, ep_len, ep_rU, ep_rH, ep_rP, ep_cV, ep_cD, ep_cH = 0,0,0,0,0,0,0,0,0,0
        u_g = np.zeros((n_gen, n_T))
        ROCOF_t, f_Nadir_t, f_Qss_t, curtailment_t, shedding_t = (np.zeros(n_T) for _ in range(5))
        
        for i in range(max_ep_len):
            o2, r, c1, c2, _, reward_u, reward_HESS, reward_p, cost_vio, cost_deta, cost_hess,\
               ROCOF, f_Nadir, f_Qss, curtailment, shedding, u_state = test_env.step(get_action(o, deterministic), o, ep_len, Train=Train, n_test_sce=n_test_sce)
                
            ep_ret, ep_cost1, ep_cost2, ep_len = ep_ret+r, ep_cost1+c1, ep_cost2+c2, ep_len+1
            ep_rU, ep_rH, ep_rP = ep_rU+reward_u, ep_rH+reward_HESS, ep_rP+reward_p
            ep_cV, ep_cD, ep_cH = ep_cV+cost_vio, ep_cD+cost_deta, ep_cH+cost_hess
            
            u_g[:, i] = u_state
            ROCOF_t[i], f_Nadir_t[i], f_Qss_t[i] = ROCOF, f_Nadir, f_Qss
            curtailment_t[i], shedding_t[i] = curtailment, shedding
            o = o2
            
        return ep_ret, ep_cost1, ep_cost2, ep_len, ep_rU, ep_rH, ep_rP, ep_cV, ep_cD, ep_cH, u_g, ROCOF_t, f_Nadir_t, f_Qss_t, curtailment_t, shedding_t
                
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)

    # ==================== 测试阶段 ====================
    if not Train:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(f'./saved_models/Case39_{args.Model_version}'))
        
        num_wind, n_test_sce = 0, 1000
        start = time.time()
            
        ep_ret, ep_cost1, ep_cost2, ep_len, ep_rU, ep_rH, ep_rP, ep_cV, ep_cD, ep_cH, \
            u_g, ROCOF_t, f_Nadir_t, f_Qss_t, curtailment_t, shedding_t = test_agent(num_wind=num_wind, deterministic=True, n_test_sce=n_test_sce)
            
        
        print("求解时间:%.4f秒" % (time.time() - start))
        print('\n==================Test Result===================')
        print(f'     The value of Ep_ret      |   {ep_ret:.2f}')
        print(f'     The value of Ep_cost1    |   {ep_cost1:.2f}')
        print(f'     The value of Ep_cost2    |   {ep_cost2:.2f}')
        print('----------------------             ---------------')
        print(f'       The Start_up cost      |   {ep_rU:.2f}')
        print(f'       The operating cost     |   {ep_rU+ep_rP:.2f}')
        print(f'      Total cost for HESS     |   {ep_rU+ep_rP+ep_rH:.2f}')
        print(f'  Expected energy curtailment |   {np.sum(curtailment_t):.2f} MWh')
        print(f'    Expected Load Shedding    |   {np.sum(shedding_t):.2f} MWh')
        print('=================================================')
        
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.plot(range(n_T), ROCOF_t); ax1.set_title('ROCOF')
        ax2.plot(range(n_T), f_Nadir_t); ax2.set_title('f_Nadir')
        sns.heatmap(u_g, linewidths=0.05, linecolor='red', annot=True, ax=ax3)
        
        if args.SaveFig:
            now = datetime.datetime.now()
            savefname = f'./Debug_record/{now.year}.{now.month}.{now.day}/Figture'
            os.makedirs(savefname, exist_ok=True)
            plt.savefig(f'{savefname}/Output result_{num_wind}_{args.Model_version}.png', dpi=500)
        plt.show()
            
    # ==================== 训练阶段 ====================
    else:
        writer = tf.summary.FileWriter(".logs/", sess.graph)
        if Continue_Training:
            if Model_Portion:
                sess.run(tf.global_variables_initializer())
                sess.run(get_target_update('main', 'target', 0.0))
                sess.run(sync_all_params())
                Loading_model = [get_vars('main/q'), get_vars('target'), get_vars('costpen'), get_vars('entreg')]
                tf.train.Saver(Loading_model).restore(sess, tf.train.latest_checkpoint(f'./saved_models/Case39_{args.Model_version}'))
            else:
                tf.train.Saver().restore(sess, tf.train.latest_checkpoint(f'./saved_models/Case39_{args.Model_version}'))
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(get_target_update('main', 'target', 0.0))
            sess.run(sync_all_params())
            
        logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                              outputs={'mu': mu, 'pi': pi, 'qr1': qr1, 'qr2': qr2, 'qc1': qc1, 'qc2': qc2})
    
        start_time = time.time()
        o, r, d, ep_ret, ep_cost1, ep_cost2, ep_len = env.reset(), 0, False, 0, 0, 0, 0
        total_steps = steps_per_epoch * epochs * max_ep_len
    
        vars_to_get = dict(LossPi=pi_loss, LossQR1=qr1_loss, LossQR2=qr2_loss, 
                           LossQC1=qc1_loss, LossQC1Var=qc1_var_loss, LossQC2=qc2_loss, LossQC2Var=qc2_var_loss,
                           QR1Vals=qr1, QR2Vals=qr2, QC1Vals=qc1, QC1Var=qc1_var, QC2Vals=qc2, QC2Var=qc2_var,
                           LogPi=logp_pi, PiEntropy=pi_entropy, Alpha=alpha, LogAlpha=log_alpha, LossAlpha=alpha_loss)
                           
        if use_costs:
            vars_to_get.update(dict(Beta1=beta1, LogBeta1=log_beta1, LossBeta1=beta1_loss, Beta2=beta2, LogBeta2=log_beta2, LossBeta2=beta2_loss))
    
        cum_cost1, cum_cost2, local_steps = 0, 0, 0
        local_steps_per_epoch = steps_per_epoch // num_procs()
        local_batch_size = batch_size // num_procs()
        epoch_start_time = time.time()
        
        for t in range(total_steps // num_procs()):
            
            # --- 分段学习率计算逻辑 ---
            current_epoch = t // (local_steps_per_epoch * max_ep_len)
            if current_epoch < 4000:
                current_lr = 1e-3
            else:
                current_lr = 1e-4

            a = get_action(o) if t > local_start_steps else env.action_sample()
    
            o2, r, c1, c2, d = env.step(a, o, ep_len)
            r *= reward_scale 
            ep_ret, ep_cost1, ep_cost2, ep_len, local_steps = ep_ret+r, ep_cost1+c1, ep_cost2+c2, ep_len+1, local_steps+1
            cum_cost1 += c1
            cum_cost2 += c2
    
            replay_buffer.store(o, a, r, o2, d, c1, c2)
            o = o2
            
            if d or (ep_len == max_ep_len):
                logger.store(EpRet=ep_ret, EpCost1=ep_cost1, EpCost2=ep_cost2)
                o, r, d, ep_ret, ep_cost1, ep_cost2, ep_len = env.reset(n=np.random.randint(n_wind_data)), 0, False, 0, 0, 0, 0
                             
            if t > 0 and t % update_freq == 0:
                for j in range(update_freq):
                    batch = replay_buffer.sample_batch(local_batch_size)
                    # 动态喂入当代的学习率 lr_ph
                    feed_dict = {x_ph: batch['obs1'], x2_ph: batch['obs2'], a_ph: batch['acts'],
                                 r_ph: batch['rews'], c1_ph: batch['costs1'], c2_ph: batch['costs2'], 
                                 d_ph: batch['done'], lr_ph: current_lr}
                                 
                    values = sess.run(vars_to_get, feed_dict) if t < local_update_after else sess.run([vars_to_get, grouped_update], feed_dict)[0]
                    logger.store(**values)
                        
                ETA = int((time.time() - epoch_start_time) * (total_steps // update_freq - t // update_freq))
                print(f'Training: [epoch:{t // (local_steps_per_epoch * max_ep_len)}|{epochs-1}] '
                      f'[batch:{(t % (local_steps_per_epoch * max_ep_len) // max_ep_len)}|{steps_per_epoch-1}] '
                      f'ETA {ETA//3600}:{ETA%3600//60}:{ETA%60} | LR: {current_lr}')
                epoch_start_time = time.time() 
                
            if t > 0 and t % (local_steps_per_epoch * max_ep_len) == 0:
                epoch = t // (local_steps_per_epoch * max_ep_len)
                cost_rate1 = mpi_sum(cum_cost1) / ((epoch + 1) * steps_per_epoch)
                cost_rate2 = mpi_sum(cum_cost2) / ((epoch + 1) * steps_per_epoch)
      
                if (epoch % save_freq == 0) or (epoch == epochs - 1):
                    saver = tf.train.Saver()
                    saver.save(sess, f'./saved_models/Case39_{epoch}/Model')
                    try:
                        now_time = datetime.datetime.now()
                        path_debug = f'./Debug_record/{now_time.year}.{now_time.month}.{now_time.day}'
                        os.makedirs(path_debug, exist_ok=True)
                        saver.save(sess, f'{path_debug}/Case39_{epoch}/Model')
                        
                        path_xls = f'{path_debug}/ReplayBuffer'
                        os.makedirs(path_xls, exist_ok=True)
                        replay_buffer.savebuffer('./logger/Buffer.xlsx')
                        replay_buffer.savebuffer(f'{path_xls}/Buffer_{now_time.hour}.xlsx')
                    except Exception as e:
                        print('保存路径出错:', e)

                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('EpRet', with_min_and_max=True) 
                logger.log_tabular('EpCost1', with_min_and_max=True)
                logger.log_tabular('EpCost2', with_min_and_max=True)
                logger.log_tabular('CostRate1', cost_rate1)
                logger.log_tabular('CostRate2', cost_rate2)
                logger.log_tabular('LossPi', with_min_and_max=True)
                logger.log_tabular('LossQR1', with_min_and_max=True)
                logger.log_tabular('LossQC1', with_min_and_max=True)
                logger.log_tabular('LossQC1Var', with_min_and_max=True)
                logger.log_tabular('LossQC2', with_min_and_max=True)
                logger.log_tabular('LossQC2Var', with_min_and_max=True)
                logger.log_tabular('PiEntropy', with_min_and_max=True)
                logger.log_tabular('TotalTime', time.time() - start_time)
                logger.dump_tabular()
        writer.close()
        
if __name__ == '__main__':
    mpi_fork(args.cpu)
    logger_kwargs = args.logger_kwargs_str if args.logger_kwargs_str else setup_logger_kwargs(args.exp_name)
    
    BC_RSAC(lambda: unPower(uncertain=True, n_sce=5), actor_fn=mlp_actor, critic_fn=mlp_critic, var_fn=mlp_var,
        ac_kwargs_actor=dict(hidden_sizes=args.hidden_sizes_actor, cnn_sizes=args.cnn_actor),
        ac_kwargs_critic=dict(hidden_sizes=args.hidden_sizes_critic, cnn_sizes=args.cnn_critic),
        ac_kwargs_var=dict(hidden_sizes=args.hidden_sizes_var, cnn_sizes=args.cnn_var), 
        gamma=args.gamma, cl=args.cl, seed=args.seed, epochs=args.epochs, replay_size=24000, batch_size=args.batch_size,
        logger_kwargs=logger_kwargs, steps_per_epoch=args.steps_per_epoch,
        update_freq=args.update_freq, render=args.render,
        local_start_steps=args.local_start_steps, save_freq=args.save_freq, local_update_after=args.local_update_after,
        fixed_entropy_bonus=args.fixed_entropy_bonus, entropy_constraint=args.entropy_constraint,
        fixed_cost_penalty1=args.fixed_cost_penalty1, cost_constraint1=args.cost_constraint1, cost_lim1=args.cost_lim1,
        fixed_cost_penalty2=args.fixed_cost_penalty2, cost_constraint2=args.cost_constraint2, cost_lim2=args.cost_lim2,
        lr_scale=args.lr_s, damp_scale=args.damp_s, Train=args.Train, Continue_Training=args.Continue,
        Model_Portion=args.Portion, load_buffer=args.load_buffer
        )
    
    if args.Train:
        Path_txt, Path_xls = './logger/progress.txt', './logger/progress.xls'
        logger_Dict = save_data.txt_xls(Path_txt, Path_xls, convert=True)

        Fig(logger_Dict, 'EpRet', c="#496C88", SaveFig=args.SaveFig, tsplot=True)
        Fig(logger_Dict, 'EpCost1', c="#FA7F6F", SaveFig=args.SaveFig, tsplot=True) 
        Fig(logger_Dict, 'EpCost2', c="#800080", SaveFig=args.SaveFig, tsplot=True) 
        Fig(logger_Dict, 'CostRate1', SaveFig=args.SaveFig, tsplot=False) 
        Fig(logger_Dict, 'LossPi', Title='The Loss function of action network', c="#8ECFC9", SaveFig=args.SaveFig, tsplot=True) 
        Fig(logger_Dict, 'PiEntropy', c="#FE4365", SaveFig=args.SaveFig, tsplot=True) 
        Fig(logger_Dict, 'LossQC1', c="#E6B33D", SaveFig=args.SaveFig, tsplot=True) 
        Fig(logger_Dict, 'LossQC2', c="#00BFFF", SaveFig=args.SaveFig, tsplot=True) 
        Fig(logger_Dict, 'LossQR1', c="#E6B33D", SaveFig=args.SaveFig, tsplot=True) 
        Fig(logger_Dict, 'LossQC1Var', c="#E6B33D", SaveFig=args.SaveFig, tsplot=True) 
        Fig(logger_Dict, 'LossQC2Var', c="#00BFFF", SaveFig=args.SaveFig, tsplot=True) 
        
        if args.SaveFig:
            now_time = datetime.datetime.now()
            src, des = './logger', f'./Debug_record/{now_time.year}.{now_time.month}.{now_time.day}/logger_{now_time.hour}'
            os.makedirs(des, exist_ok=True)
            for file in os.listdir(src):
                full_file_name = os.path.join(src, file)
                if os.path.isfile(full_file_name):
                    try:
                        shutil.copy(full_file_name, des)
                    except Exception as e:
                        print(f'保存文件 {file} 时出错:', e)
                        
            with open(os.path.join(des, 'args.txt'), "w", encoding="utf-8") as log_args:
                for k, v in vars(args).items():
                    log_args.write(f"{k}: {v}\n")