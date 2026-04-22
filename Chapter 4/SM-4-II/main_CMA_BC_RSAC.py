"""
Chapter4:CMA-BC-RSAC

智能体配置:
- 智能体1: IEEE118节点系统
- 智能体2: IEEE39节点系统

"""

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
from env4 import unPower

from context_encoder import (
    ContextEncoder, MultiAgentContextEncoder, LocalContextBuffer,
    build_context_conditioned_actor, build_context_conditioned_critic
)

from env4 import (n_T, n_wind, n_load, n_gen, n_wind_data, state_dim, action_dim,
                  n_wind1, n_wind2, n_gen1, n_gen2, n_load1, n_load2,
                  state_dim1, state_dim2, action_dim1, action_dim2, n_agent)

obs_dim1 = state_dim1
obs_dim2 = state_dim2
act_dim1 = action_dim1
act_dim2 = action_dim2
global_state_dim = state_dim
total_act_dim = act_dim1 + act_dim2

EPS = 1e-8

# 上下文参数
CONTEXT_LEN = 5
LATENT_DIM = 16

# 局部上下文输入维度: context_i = (o_i, a_i, r_i, c_i) 的历史序列
CONTEXT_INPUT_DIM1 = obs_dim1 + act_dim1 + 2
CONTEXT_INPUT_DIM2 = obs_dim2 + act_dim2 + 2

# 全局上下文维度（仅用于Critic集中式训练）
GLOBAL_CONTEXT_DIM = obs_dim1 + obs_dim2 + act_dim1 + act_dim2 + 4

# 预计算索引范围
_IDX = {
    'wind1_end': n_wind1,
    'wind1_grad_start': n_wind1,
    'wind1_grad_end': 2 * n_wind1,
    'load1_start': 2 * n_wind1,
    'load1_end': 2 * n_wind1 + n_load1,
    'u1_start': 2 * n_wind1 + n_load1,
    'u1_end': 2 * n_wind1 + n_load1 + n_gen1,
    'tao1_start': 2 * n_wind1 + n_load1 + n_gen1,
    'tao1_end': 2 * n_wind1 + n_load1 + 2 * n_gen1,
    'wind2_end': n_wind2,
    'wind2_grad_start': n_wind2,
    'wind2_grad_end': 2 * n_wind2,
    'load2_start': 2 * n_wind2,
    'load2_end': 2 * n_wind2 + n_load2,
    'u2_start': 2 * n_wind2 + n_load2,
    'u2_end': 2 * n_wind2 + n_load2 + n_gen2,
    'tao2_start': 2 * n_wind2 + n_load2 + n_gen2,
    'tao2_end': 2 * n_wind2 + n_load2 + 2 * n_gen2,
}


def build_global_state(o1, o2):
    """从局部观测构建全局状态"""
    return np.concatenate([
        o1[:_IDX['wind1_end']], o2[:_IDX['wind2_end']],
        o1[_IDX['wind1_grad_start']:_IDX['wind1_grad_end']], 
        o2[_IDX['wind2_grad_start']:_IDX['wind2_grad_end']],
        [o1[_IDX['load1_start']] + o2[_IDX['load2_start']]],
        o1[_IDX['u1_start']:_IDX['u1_end']],
        o2[_IDX['u2_start']:_IDX['u2_end']],
        o1[_IDX['tao1_start']:_IDX['tao1_end']],
        o2[_IDX['tao2_start']:_IDX['tao2_end']],
        [o1[-1], o2[-1]]
    ])


snow = datetime.datetime.now()
seed_now = int(str(snow.month) + str(snow.day) + str(snow.hour))


# ==================== 训练参数调度器 ====================
class TrainingScheduler:
    """
    四阶段调度策略：
    - 阶段1 (0-2000): 探索 - 高学习率，频繁Actor更新
    - 阶段2 (2000-6000): 稳定学习
    - 阶段3 (6000-12000): 精细调整 - 降低学习率
    - 阶段4 (12000-16000): 收敛 - 最保守设置
    """
    
    def __init__(self, total_epochs=16000):
        self.total_epochs = total_epochs
        self.phase_boundaries = [0, 2000, 4000, 8000, total_epochs]
        
        self.schedule = {
            'lr_actor': [1e-5, 5e-6, 3e-6, 1e-6],
            'lr_critic': [5e-5, 3e-5, 1e-5, 5e-6],
            'lr_context': [5e-5, 3e-5, 1e-5, 5e-6],
            'actor_update_freq': [2, 3, 4, 5],
            'grad_clip': [0.5, 0.4, 0.3, 0.2],
            'polyak': [0.995, 0.997, 0.998, 0.999],
            'entropy_constraint': [-0.5, -0.4, -0.35, -0.3],
        }
        
        self.current_phase = 0
        self.current_params = {}
        self._update_params(0)
    
    def _get_phase(self, epoch):
        for i in range(len(self.phase_boundaries) - 1):
            if self.phase_boundaries[i] <= epoch < self.phase_boundaries[i + 1]:
                return i
        return len(self.phase_boundaries) - 2
    
    def _update_params(self, epoch):
        phase = self._get_phase(epoch)
        self.current_phase = phase
        for param_name, values in self.schedule.items():
            self.current_params[param_name] = values[phase]
    
    def get_params(self, epoch):
        new_phase = self._get_phase(epoch)
        if new_phase != self.current_phase:
            self._update_params(epoch)
            print(f'\n========== 训练阶段切换: 阶段{self.current_phase + 1} (epoch {epoch}) ==========')
            print(f'  lr_actor: {self.current_params["lr_actor"]:.2e}')
            print(f'  lr_critic: {self.current_params["lr_critic"]:.2e}')
            print(f'  actor_update_freq: {self.current_params["actor_update_freq"]}')
            print(f'  grad_clip: {self.current_params["grad_clip"]}')
            print(f'  polyak: {self.current_params["polyak"]}')
            print(f'============================================================\n')
        return self.current_params.copy()
    
    def get_phase_info(self, epoch):
        phase = self._get_phase(epoch)
        phase_names = ['探索阶段', '稳定学习阶段', '精细调整阶段', '收敛阶段']
        return f"阶段{phase + 1}: {phase_names[phase]}"
    
    @staticmethod
    def linear_interpolate(epoch, start_epoch, end_epoch, start_val, end_val):
        if epoch <= start_epoch:
            return start_val
        if epoch >= end_epoch:
            return end_val
        ratio = (epoch - start_epoch) / (end_epoch - start_epoch)
        return start_val + ratio * (end_val - start_val)


# ==================== 参数解析 ====================
parser = argparse.ArgumentParser()
parser.add_argument('--Train', type=bool, default=True)
parser.add_argument('--Continue', type=bool, default=False)
parser.add_argument('--Portion', type=bool, default=False)
parser.add_argument('--SaveFig', type=bool, default=True)
parser.add_argument('--load_buffer', type=bool, default=False)
parser.add_argument('--Model_version', type=int, default=0)
parser.add_argument('--epochs', type=int, default=12001)
parser.add_argument('--steps_per_epoch', type=int, default=1)
parser.add_argument('--update_freq', type=int, default=n_T)
parser.add_argument('--save_freq', type=int, default=2000)
parser.add_argument('--cost_lim', type=float, default=1)
parser.add_argument('--cl', type=float, default=0.95)

parser.add_argument('--use_scheduler', type=bool, default=True)

# 学习率（初始值，会被调度器覆盖）
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--lr_actor', type=float, default=1e-5)
parser.add_argument('--lr_critic', type=float, default=5e-5)
parser.add_argument('--lr_decay_start', type=int, default=2000)
parser.add_argument('--lr_decay_rate', type=float, default=0.5)

# 稳定性参数（初始值，会被调度器覆盖）
parser.add_argument('--polyak', type=float, default=0.995)
parser.add_argument('--q_clip_max', type=float, default=3e4)
parser.add_argument('--grad_clip', type=float, default=0.5)
parser.add_argument('--actor_update_freq', type=int, default=2)
parser.add_argument('--min_entropy_ratio', type=float, default=0.5)

# 网络结构
parser.add_argument('--hidden_sizes_actor', type=list, default=[128, 256, 64])
parser.add_argument('--hidden_sizes_critic', type=list, default=[128, 256, 64])
parser.add_argument('--hidden_sizes_var', type=list, default=[128, 256, 64])
parser.add_argument('--cnn_actor', type=list, default=[])
parser.add_argument('--cnn_critic', type=list, default=[])
parser.add_argument('--cnn_var', type=list, default=[])

# 上下文编码器
parser.add_argument('--context_len', type=int, default=5)
parser.add_argument('--latent_dim', type=int, default=16)
parser.add_argument('--encoder_hidden', type=list, default=[128, 64])
parser.add_argument('--decoder_hidden', type=list, default=[64, 128])
parser.add_argument('--kl_weight', type=float, default=0.01)
parser.add_argument('--recon_weight', type=float, default=1.0)

parser.add_argument('--hid', type=int, default=256)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=seed_now)
parser.add_argument('--exp_name', type=str, default='cma_bc_rsac')
parser.add_argument('--cpu', type=int, default=1)
parser.add_argument('--render', default=False, action='store_true')
parser.add_argument('--local_start_steps', default=n_T*100, type=int)
parser.add_argument('--local_update_after', default=n_T*100, type=int)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--fixed_entropy_bonus', default=None, type=float)
parser.add_argument('--entropy_constraint', type=float, default=-0.5)
parser.add_argument('--fixed_cost_penalty', default=None, type=float)
parser.add_argument('--cost_constraint', type=float, default=None)
parser.add_argument('--lr_s', type=int, default=50)
parser.add_argument('--damp_s', type=int, default=10)
parser.add_argument('--logger_kwargs_str', type=json.loads, default='{"output_dir": "./logger"}')
args = parser.parse_args()


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def cnn_mlp(x, hidden_sizes=(64,), activation=tf.tanh, output_activation=None, cnn_sizes=[]):
    j = 1
    if cnn_sizes != []:
        dim = int(x.shape[1])
        Img_shape = (dim, 1)
        x = tf.reshape(x, (-1, *Img_shape))
        for i in cnn_sizes:
            x = tf.layers.conv1d(
                inputs=x, filters=i, kernel_size=3,
                padding='same', use_bias=False,
                activation=None, name='cnn_%d' % j)
            x = tf.layers.batch_normalization(inputs=x, training=True, name='bn_%d' % j)
            x = tf.tanh(x)
            j += 1
        x = tf.reshape(x, (-1, dim * cnn_sizes[-1]))
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation, name='fc_%d' % j)
        j += 1
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, name='fc_%d' % j)


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def get_target_update(main_name, target_name, polyak):
    main_vars = {x.name: x for x in get_vars(main_name)}
    targ_vars = {x.name: x for x in get_vars(target_name)}
    assign_ops = []
    for v_targ in targ_vars:
        assert v_targ.startswith(target_name), f'bad var name {v_targ} for {target_name}'
        v_main = v_targ.replace(target_name, main_name, 1)
        assert v_main in main_vars, f'missing var name {v_main}'
        assign_op = tf.assign(targ_vars[v_targ], polyak * targ_vars[v_targ] + (1 - polyak) * main_vars[v_main])
        assign_ops.append(assign_op)
    return tf.group(assign_ops)


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
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


def mlp_context_actor(x, z, a, name='pi', hidden_sizes=(64, 64), activation=tf.nn.relu,
                      output_activation=None, cnn_sizes=[]):
    """上下文条件Actor - 分布式执行（局部观测 + 局部隐变量z）"""
    act_dim = a.shape.as_list()[-1]
    
    with tf.variable_scope(name):
        x_concat = tf.concat([x, z], axis=-1)
        net = cnn_mlp(x_concat, list(hidden_sizes), activation, activation, list(cnn_sizes))
        mu = tf.layers.dense(net, act_dim, activation=output_activation, name='fc_mu')
        log_std = tf.layers.dense(net, act_dim, activation=None, name='fc_std')
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        std = tf.exp(log_std)
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp_pi = gaussian_likelihood(pi, mu, log_std)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
        
    return mu, pi, logp_pi


def mlp_context_critic(x, a, pi, z, name, hidden_sizes=(64, 64), activation=tf.nn.relu,
                       output_activation=None, cnn_sizes=[]):
    """上下文条件Critic - 集中式训练（全局状态 + 联合动作 + 全局隐变量z）"""
    fn_mlp = lambda inp: tf.squeeze(cnn_mlp(x=inp,
                                            hidden_sizes=list(hidden_sizes) + [1],
                                            activation=activation,
                                            output_activation=None,
                                            cnn_sizes=cnn_sizes),
                                    axis=1)
    with tf.variable_scope(name):
        critic = fn_mlp(tf.concat([x, a, z], axis=-1))

    with tf.variable_scope(name, reuse=True):
        critic_pi = fn_mlp(tf.concat([x, pi, z], axis=-1))

    return critic, critic_pi


def mlp_context_var(x, a, pi, z, name, hidden_sizes=(64, 64), activation=tf.nn.relu,
                    output_activation=None, cnn_sizes=[]):
    """上下文条件方差网络 - 用于CVaR约束"""
    fn_mlp = lambda inp: tf.squeeze(cnn_mlp(x=inp,
                                            hidden_sizes=list(hidden_sizes) + [1],
                                            activation=activation,
                                            output_activation=None,
                                            cnn_sizes=cnn_sizes),
                                    axis=1)
    with tf.variable_scope(name):
        var = fn_mlp(tf.concat([x, a, z], axis=-1))
        var = tf.nn.softplus(var)

    with tf.variable_scope(name, reuse=True):
        var_pi = fn_mlp(tf.concat([x, pi, z], axis=-1))
        var_pi = tf.nn.softplus(var_pi)

    return var, var_pi


def Fig(logger_Dict, item='EpCost', c=None, Title=None, SaveFig=True, tsplot=True):
    try:
        epoch_data = np.array(logger_Dict['Epoch'])
        title = Title if Title else f'The Expectations of {item} of Each Epoch'
        
        if tsplot:
            data = np.concatenate([
                np.array(logger_Dict['Average' + item]),
                np.array(logger_Dict['Min' + item]),
                np.array(logger_Dict['Max' + item])
            ])
            epoch = np.tile(epoch_data, 3)
            sns.set(style="whitegrid")
            sns.lineplot(x=epoch, y=data, color=c, lw=0.8)
        else:
            plt.plot(epoch_data, logger_Dict[item])
        
        plt.xlabel('Epoch')
        plt.ylabel(item)
        plt.title(title)
        
        if SaveFig:
            now = datetime.datetime.now()
            savefname = f'./Debug_record/{now.year}.{now.month}.{now.day}/Figture'
            os.makedirs(savefname, exist_ok=True)
            plt.savefig(f'{savefname}/{now.hour}-{now.minute}-{item}.png', dpi=500)
        plt.show()
    except Exception as e:
        print(f'绘图失败 "{item}": {e}')


class MultiAgentContextReplayBuffer:
    """
    多智能体上下文感知经验回放 - 符合CTDE原则
    
    时序关系：
    - t时刻决策前：获取 context_{t-1}（历史 t-H 到 t-1）
    - t时刻选择动作：a_t = π(o_t | z_t)，z_t 来自 context_{t-1}
    - 存储经验：(o_t, a_t, r_t, c_t, o_{t+1}, context_{t-1})
    - t+1时刻开始时：将 (o_t, a_t, r_t, c_t) 加入历史
    """

    def __init__(self, obs_dim1, obs_dim2, act_dim1, act_dim2, global_state_dim, 
                 context_dim1, context_dim2, context_len, size, load_buffer=False, 
                 Path_xls='./logger/MABuffer.xlsx'):
        self.obs1_buf = np.zeros([size, obs_dim1], dtype=np.float32)
        self.obs1_next_buf = np.zeros([size, obs_dim1], dtype=np.float32)
        self.acts1_buf = np.zeros([size, act_dim1], dtype=np.float32)
        self.rews1_buf = np.zeros(size, dtype=np.float32)
        self.costs1_buf = np.zeros(size, dtype=np.float32)

        self.obs2_buf = np.zeros([size, obs_dim2], dtype=np.float32)
        self.obs2_next_buf = np.zeros([size, obs_dim2], dtype=np.float32)
        self.acts2_buf = np.zeros([size, act_dim2], dtype=np.float32)
        self.rews2_buf = np.zeros(size, dtype=np.float32)
        self.costs2_buf = np.zeros(size, dtype=np.float32)

        self.global_state_buf = np.zeros([size, global_state_dim], dtype=np.float32)
        self.global_state_next_buf = np.zeros([size, global_state_dim], dtype=np.float32)

        self.context1_buf = np.zeros([size, context_len, context_dim1], dtype=np.float32)
        self.context2_buf = np.zeros([size, context_len, context_dim2], dtype=np.float32)
        
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size
        self.context_len = context_len
        self.context_dim1 = context_dim1
        self.context_dim2 = context_dim2
        self.obs_dim1 = obs_dim1
        self.obs_dim2 = obs_dim2
        self.act_dim1 = act_dim1
        self.act_dim2 = act_dim2
        
        self._local_context1 = LocalContextBuffer(context_len, obs_dim1, act_dim1)
        self._local_context2 = LocalContextBuffer(context_len, obs_dim2, act_dim2)

        if load_buffer:
            self._load_buffer(Path_xls, size)

    def _load_buffer(self, Path_xls, size):
        try:
            data = pd.read_excel(Path_xls, sheet_name=None)
            u = min(size, len(data['obs1']))
            
            buf_2d_map = {
                'obs1': self.obs1_buf, 'obs1_next': self.obs1_next_buf,
                'acts1': self.acts1_buf, 'obs2': self.obs2_buf,
                'obs2_next': self.obs2_next_buf, 'acts2': self.acts2_buf,
                'global_state': self.global_state_buf,
                'global_state_next': self.global_state_next_buf
            }
            for key, buf in buf_2d_map.items():
                if key in data:
                    buf[:u] = np.array(data[key])[:u]
            
            buf_1d_map = {
                'rews1': self.rews1_buf, 'costs1': self.costs1_buf,
                'rews2': self.rews2_buf, 'costs2': self.costs2_buf,
                'done': self.done_buf
            }
            for key, buf in buf_1d_map.items():
                if key in data:
                    buf[:u] = np.array(data[key])[:u].ravel()

            self.size = u
            self.ptr = u % self.max_size
            print('多智能体上下文经验池已加载\n')
        except Exception as e:
            print(f'经验池加载失败: {e}')

    def get_context_for_decision(self):
        """获取决策前的局部上下文（t-H到t-1），严格遵循时间因果性"""
        context1 = self._local_context1.get_context_for_decision()
        context2 = self._local_context2.get_context_for_decision()
        return context1, context2

    def add_transition_to_history(self, obs1, act1, rew1, cost1, 
                                   obs2, act2, rew2, cost2):
        """将当前步转移加入历史（step之后、下一次决策之前调用）"""
        self._local_context1.add_transition(obs1, act1, rew1, cost1)
        self._local_context2.add_transition(obs2, act2, rew2, cost2)

    def reset_context(self):
        self._local_context1.reset()
        self._local_context2.reset()

    def store(self, obs1, obs2, act1, act2, rew1, rew2, obs1_next, obs2_next,
              global_state, global_state_next, done, cost1, cost2, 
              context1, context2):
        """存储经验，context为决策前获取的上下文"""
        self.obs1_buf[self.ptr] = obs1
        self.obs1_next_buf[self.ptr] = obs1_next
        self.acts1_buf[self.ptr] = act1
        self.rews1_buf[self.ptr] = rew1
        self.costs1_buf[self.ptr] = cost1

        self.obs2_buf[self.ptr] = obs2
        self.obs2_next_buf[self.ptr] = obs2_next
        self.acts2_buf[self.ptr] = act2
        self.rews2_buf[self.ptr] = rew2
        self.costs2_buf[self.ptr] = cost2

        self.global_state_buf[self.ptr] = global_state
        self.global_state_next_buf[self.ptr] = global_state_next
        
        self.context1_buf[self.ptr] = context1
        self.context2_buf[self.ptr] = context2
        
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs1_next=self.obs1_next_buf[idxs],
            acts1=self.acts1_buf[idxs],
            rews1=self.rews1_buf[idxs],
            costs1=self.costs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            obs2_next=self.obs2_next_buf[idxs],
            acts2=self.acts2_buf[idxs],
            rews2=self.rews2_buf[idxs],
            costs2=self.costs2_buf[idxs],
            global_state=self.global_state_buf[idxs],
            global_state_next=self.global_state_next_buf[idxs],
            context1=self.context1_buf[idxs],
            context2=self.context2_buf[idxs],
            done=self.done_buf[idxs]
        )

    def savebuffer(self, Path_xls):
        data = dict(
            obs1=self.obs1_buf,
            obs1_next=self.obs1_next_buf,
            acts1=self.acts1_buf,
            rews1=self.rews1_buf,
            costs1=self.costs1_buf,
            obs2=self.obs2_buf,
            obs2_next=self.obs2_next_buf,
            acts2=self.acts2_buf,
            rews2=self.rews2_buf,
            costs2=self.costs2_buf,
            global_state=self.global_state_buf,
            global_state_next=self.global_state_next_buf,
            done=self.done_buf
        )
        with pd.ExcelWriter(Path_xls) as writer:
            for key in data:
                df = pd.DataFrame(data[key])
                df.to_excel(writer, sheet_name=key, index=False)


def cma_bc_rsac(env_fn, 
             ac_kwargs_actor=dict(), ac_kwargs_critic=dict(), ac_kwargs_var=dict(),
             seed=0, steps_per_epoch=1, epochs=16001, replay_size=24000, gamma=0.99, cl=0.5,
             polyak=0.995, lr=5e-5, lr_actor=1e-5, lr_critic=5e-5,
             batch_size=512, local_start_steps=600,
             max_ep_len=n_T, logger_kwargs=dict(), save_freq=2000, local_update_after=int(1e3),
             update_freq=24, render=False,
             fixed_entropy_bonus=None, entropy_constraint=-0.5,
             fixed_cost_penalty=None, cost_constraint=None, cost_lim=None,
             reward_scale=1, lr_scale=1, damp_scale=0, Train=True, Continue_Training=False,
             Model_Portion=False, load_buffer=False,
             context_len=5, latent_dim=16, encoder_hidden=[128, 64], decoder_hidden=[64, 128],
             kl_weight=0.01, recon_weight=1.0,
             q_clip_max=3e4, grad_clip=0.5,
             actor_update_freq=2,
             lr_decay_start=2000, lr_decay_rate=0.5,
             min_entropy_ratio=0.5,
             use_scheduler=True
             ):
    """CMA-BC-RSAC"""

    use_costs = fixed_cost_penalty or cost_constraint or cost_lim

    pdf_cdf = cl ** (-1) * norm.pdf(norm.ppf(cl))

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env, test_env = env_fn(), env_fn()

    seed += 200 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # 局部上下文输入维度: (o_i, a_i, r_i, c_i)
    context_input_dim1 = obs_dim1 + act_dim1 + 2
    context_input_dim2 = obs_dim2 + act_dim2 + 2

    # ========== 占位符 ==========
    o1_ph = placeholder(obs_dim1)
    o1_next_ph = placeholder(obs_dim1)
    a1_ph = placeholder(act_dim1)
    r1_ph = placeholder()
    c1_ph = placeholder()

    o2_ph = placeholder(obs_dim2)
    o2_next_ph = placeholder(obs_dim2)
    a2_ph = placeholder(act_dim2)
    r2_ph = placeholder()
    c2_ph = placeholder()

    # 全局状态（仅用于Critic集中式训练）
    s_ph = placeholder(global_state_dim)
    s_next_ph = placeholder(global_state_dim)

    d_ph = placeholder()

    # 局部上下文占位符
    context1_ph = tf.placeholder(tf.float32, [None, context_len, context_input_dim1], name='context1')
    context2_ph = tf.placeholder(tf.float32, [None, context_len, context_input_dim2], name='context2')

    # ========== 上下文编码器（每个智能体独立） ==========
    context_encoder1 = ContextEncoder(
        context_input_dim=context_input_dim1,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        name='context_encoder1'
    )
    
    context_encoder2 = ContextEncoder(
        context_input_dim=context_input_dim2,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        name='context_encoder2'
    )

    # 推断局部隐变量
    z1_mu, z1_log_var, z1 = context_encoder1.build_encoder(context1_ph, reuse=False)
    z2_mu, z2_log_var, z2 = context_encoder2.build_encoder(context2_ph, reuse=False)
    
    # 全局上下文（仅用于Critic）- 拼接两个局部隐变量
    z_global = tf.concat([z1, z2], axis=-1)

    # 解码器 - 预测自身的奖励和成本
    pred_reward_cost1 = context_encoder1.build_decoder(z1, target_dim=2, reuse=False)
    target_reward_cost1 = tf.stack([r1_ph, c1_ph], axis=1)
    
    pred_reward_cost2 = context_encoder2.build_decoder(z2, target_dim=2, reuse=False)
    target_reward_cost2 = tf.stack([r2_ph, c2_ph], axis=1)

    # 上下文编码器损失
    kl_loss1 = context_encoder1.compute_kl_loss(z1_mu, z1_log_var)
    recon_loss1 = context_encoder1.compute_reconstruction_loss(target_reward_cost1, pred_reward_cost1)
    context_loss1 = kl_weight * kl_loss1 + recon_weight * recon_loss1
    
    kl_loss2 = context_encoder2.compute_kl_loss(z2_mu, z2_log_var)
    recon_loss2 = context_encoder2.compute_reconstruction_loss(target_reward_cost2, pred_reward_cost2)
    context_loss2 = kl_weight * kl_loss2 + recon_weight * recon_loss2
    
    kl_loss = kl_loss1 + kl_loss2
    recon_loss = recon_loss1 + recon_loss2
    context_loss = context_loss1 + context_loss2

    # ========== 主网络 ==========
    with tf.variable_scope('main'):
        # Actor: 局部观测 + 局部隐变量（CTDE分布式执行）
        mu1, pi1, logp_pi1 = mlp_context_actor(o1_ph, z1, a1_ph, name='pi1', **ac_kwargs_actor)
        mu2, pi2, logp_pi2 = mlp_context_actor(o2_ph, z2, a2_ph, name='pi2', **ac_kwargs_actor)

        a_joint = tf.concat([a1_ph, a2_ph], axis=-1)
        pi_joint = tf.concat([pi1, pi2], axis=-1)

        # Critic: 全局状态 + 联合动作 + 全局隐变量（CTDE集中式训练）
        qr1_1, qr1_1_pi = mlp_context_critic(s_ph, a_joint, pi_joint, z_global, name='qr1_1', **ac_kwargs_critic)
        qr2_1, qr2_1_pi = mlp_context_critic(s_ph, a_joint, pi_joint, z_global, name='qr2_1', **ac_kwargs_critic)
        qc1, qc1_pi = mlp_context_critic(s_ph, a_joint, pi_joint, z_global, name='qc1', **ac_kwargs_critic)
        qc1_var, qc1_pi_var = mlp_context_var(s_ph, a_joint, pi_joint, z_global, name='qc1_var', **ac_kwargs_var)

        qr1_2, qr1_2_pi = mlp_context_critic(s_ph, a_joint, pi_joint, z_global, name='qr1_2', **ac_kwargs_critic)
        qr2_2, qr2_2_pi = mlp_context_critic(s_ph, a_joint, pi_joint, z_global, name='qr2_2', **ac_kwargs_critic)
        qc2, qc2_pi = mlp_context_critic(s_ph, a_joint, pi_joint, z_global, name='qc2', **ac_kwargs_critic)
        qc2_var, qc2_pi_var = mlp_context_var(s_ph, a_joint, pi_joint, z_global, name='qc2_var', **ac_kwargs_var)

    # 下一时刻的策略输出
    with tf.variable_scope('main', reuse=True):
        _, pi1_next, logp_pi1_next = mlp_context_actor(o1_next_ph, z1, a1_ph, name='pi1', **ac_kwargs_actor)
        _, pi2_next, logp_pi2_next = mlp_context_actor(o2_next_ph, z2, a2_ph, name='pi2', **ac_kwargs_actor)
        pi_joint_next = tf.concat([pi1_next, pi2_next], axis=-1)

    # ========== 目标网络 ==========
    with tf.variable_scope('target'):
        _, qr1_1_pi_targ = mlp_context_critic(s_next_ph, a_joint, pi_joint_next, z_global, name='qr1_1', **ac_kwargs_critic)
        _, qr2_1_pi_targ = mlp_context_critic(s_next_ph, a_joint, pi_joint_next, z_global, name='qr2_1', **ac_kwargs_critic)
        _, qc1_pi_targ = mlp_context_critic(s_next_ph, a_joint, pi_joint_next, z_global, name='qc1', **ac_kwargs_critic)
        _, qc1_pi_var_targ = mlp_context_var(s_next_ph, a_joint, pi_joint_next, z_global, name='qc1_var', **ac_kwargs_var)

        _, qr1_2_pi_targ = mlp_context_critic(s_next_ph, a_joint, pi_joint_next, z_global, name='qr1_2', **ac_kwargs_critic)
        _, qr2_2_pi_targ = mlp_context_critic(s_next_ph, a_joint, pi_joint_next, z_global, name='qr2_2', **ac_kwargs_critic)
        _, qc2_pi_targ = mlp_context_critic(s_next_ph, a_joint, pi_joint_next, z_global, name='qc2', **ac_kwargs_critic)
        _, qc2_pi_var_targ = mlp_context_var(s_next_ph, a_joint, pi_joint_next, z_global, name='qc2_var', **ac_kwargs_var)

    # ========== 熵系数 ==========
    if fixed_entropy_bonus is None:
        with tf.variable_scope('entreg'):
            soft_alpha1 = tf.get_variable('soft_alpha1', initializer=0.0, trainable=True, dtype=tf.float32)
            soft_alpha2 = tf.get_variable('soft_alpha2', initializer=0.0, trainable=True, dtype=tf.float32)
        alpha1 = tf.nn.softplus(soft_alpha1)
        alpha2 = tf.nn.softplus(soft_alpha2)
    else:
        alpha1 = tf.constant(fixed_entropy_bonus)
        alpha2 = tf.constant(fixed_entropy_bonus)
    log_alpha1 = tf.log(tf.clip_by_value(alpha1, 1e-8, 1e8))
    log_alpha2 = tf.log(tf.clip_by_value(alpha2, 1e-8, 1e8))

    # ========== 成本惩罚系数 ==========
    if use_costs:
        if fixed_cost_penalty is None:
            with tf.variable_scope('costpen'):
                soft_beta1 = tf.get_variable('soft_beta1', initializer=0.0, trainable=True, dtype=tf.float32)
                soft_beta2 = tf.get_variable('soft_beta2', initializer=0.0, trainable=True, dtype=tf.float32)
            beta1 = tf.nn.softplus(soft_beta1)
            beta2 = tf.nn.softplus(soft_beta2)
            log_beta1 = tf.log(tf.clip_by_value(beta1, 1e-8, 1e8))
            log_beta2 = tf.log(tf.clip_by_value(beta2, 1e-8, 1e8))
        else:
            beta1 = tf.constant(fixed_cost_penalty)
            beta2 = tf.constant(fixed_cost_penalty)
            log_beta1 = tf.log(tf.clip_by_value(beta1, 1e-8, 1e8))
            log_beta2 = tf.log(tf.clip_by_value(beta2, 1e-8, 1e8))
    else:
        beta1, beta2 = 0.0, 0.0
        print('Not using costs')

    # ========== 经验回放缓冲区 ==========
    replay_buffer = MultiAgentContextReplayBuffer(
        obs_dim1=obs_dim1, obs_dim2=obs_dim2,
        act_dim1=act_dim1, act_dim2=act_dim2,
        global_state_dim=global_state_dim,
        context_dim1=context_input_dim1,
        context_dim2=context_input_dim2,
        context_len=context_len,
        size=replay_size, load_buffer=load_buffer
    )

    if proc_id() == 0:
        var_counts = tuple(count_vars(scope) for scope in
                           ['main/pi1', 'main/pi2', 'main/qr1_1', 'main/qc1', 
                            'context_encoder1', 'context_encoder2', 'main'])
        print('\nNumber of parameters: \t pi1: %d, \t pi2: %d, \t qr1_1: %d, \t qc1: %d, \t context_enc1: %d, \t context_enc2: %d, \t total: %d\n' % var_counts)

    # ========== Double Q + Q值裁剪（防止Q值发散） ==========
    min_q1_pi = tf.minimum(qr1_1_pi, qr2_1_pi)
    min_q1_pi_targ = tf.minimum(qr1_1_pi_targ, qr2_1_pi_targ)
    min_q2_pi = tf.minimum(qr1_2_pi, qr2_2_pi)
    min_q2_pi_targ = tf.minimum(qr1_2_pi_targ, qr2_2_pi_targ)
    
    min_q1_pi = tf.clip_by_value(min_q1_pi, -q_clip_max, q_clip_max)
    min_q1_pi_targ = tf.clip_by_value(min_q1_pi_targ, -q_clip_max, q_clip_max)
    min_q2_pi = tf.clip_by_value(min_q2_pi, -q_clip_max, q_clip_max)
    min_q2_pi_targ = tf.clip_by_value(min_q2_pi_targ, -q_clip_max, q_clip_max)
    
    qc1_pi = tf.clip_by_value(qc1_pi, -q_clip_max, q_clip_max)
    qc1_pi_targ = tf.clip_by_value(qc1_pi_targ, -q_clip_max, q_clip_max)
    qc2_pi = tf.clip_by_value(qc2_pi, -q_clip_max, q_clip_max)
    qc2_pi_targ = tf.clip_by_value(qc2_pi_targ, -q_clip_max, q_clip_max)

    qc1_var = tf.clip_by_value(qc1_var, 1e-8, 1e8)
    qc1_pi_var = tf.clip_by_value(qc1_pi_var, 1e-8, 1e8)
    qc1_pi_var_targ = tf.clip_by_value(qc1_pi_var_targ, 1e-8, 1e8)
    qc2_var = tf.clip_by_value(qc2_var, 1e-8, 1e8)
    qc2_pi_var = tf.clip_by_value(qc2_pi_var, 1e-8, 1e8)
    qc2_pi_var_targ = tf.clip_by_value(qc2_pi_var_targ, 1e-8, 1e8)

    # ========== 目标值计算 ==========
    q1_backup = tf.stop_gradient(r1_ph + gamma * (1 - d_ph) * (min_q1_pi_targ - alpha1 * logp_pi1_next))
    q1_backup = tf.clip_by_value(q1_backup, -q_clip_max, q_clip_max)
    
    qc1_backup = tf.stop_gradient(c1_ph + gamma * (1 - d_ph) * qc1_pi_targ)
    qc1_backup = tf.clip_by_value(qc1_backup, -q_clip_max, q_clip_max)
    
    # 方差Bellman算子：
    # V_c(s_t) = c_t^2 + γ^2*(V_c(s_{t+1}) + Q_c(s_{t+1})^2) + 2γ*c_t*Q_c(s_{t+1}) - Q_c(s_t)^2
    qc1_var_raw = (c1_ph ** 2 + 
                   2 * gamma * (1 - d_ph) * c1_ph * qc1_pi_targ +
                   gamma ** 2 * (1 - d_ph) * (qc1_pi_var_targ + qc1_pi_targ ** 2) -
                   qc1 ** 2)
    qc1_var_backup = tf.stop_gradient(tf.maximum(qc1_var_raw, 1e-6))

    q2_backup = tf.stop_gradient(r2_ph + gamma * (1 - d_ph) * (min_q2_pi_targ - alpha2 * logp_pi2_next))
    q2_backup = tf.clip_by_value(q2_backup, -q_clip_max, q_clip_max)
    
    qc2_backup = tf.stop_gradient(c2_ph + gamma * (1 - d_ph) * qc2_pi_targ)
    qc2_backup = tf.clip_by_value(qc2_backup, -q_clip_max, q_clip_max)
    
    qc2_var_raw = (c2_ph ** 2 + 
                   2 * gamma * (1 - d_ph) * c2_ph * qc2_pi_targ +
                   gamma ** 2 * (1 - d_ph) * (qc2_pi_var_targ + qc2_pi_targ ** 2) - 
                   qc2 ** 2)
    qc2_var_backup = tf.stop_gradient(tf.maximum(qc2_var_raw, 1e-6))

    # CVaR成本约束
    cost_constraint_val = cost_lim * (1 - gamma ** max_ep_len) / (1 - gamma) / max_ep_len
    damp1 = damp_scale * tf.reduce_mean(cost_constraint_val - qc1 - pdf_cdf * tf.sqrt(qc1_var))
    damp2 = damp_scale * tf.reduce_mean(cost_constraint_val - qc2 - pdf_cdf * tf.sqrt(qc2_var))

    # ========== 损失函数 ==========
    # Actor损失: 熵正则 - Q_reward + 成本惩罚*(Q_cost + CVaR)
    pi1_loss = tf.reduce_mean(alpha1 * logp_pi1 - min_q1_pi +
                              (beta1 - damp1) * (qc1_pi + pdf_cdf * tf.sqrt(qc1_pi_var)))
    pi2_loss = tf.reduce_mean(alpha2 * logp_pi2 - min_q2_pi +
                              (beta2 - damp2) * (qc2_pi + pdf_cdf * tf.sqrt(qc2_pi_var)))

    # Critic损失
    qr1_1_loss = 0.5 * tf.reduce_mean((q1_backup - qr1_1) ** 2)
    qr2_1_loss = 0.5 * tf.reduce_mean((q1_backup - qr2_1) ** 2)
    qc1_loss = 0.5 * tf.reduce_mean((qc1_backup - qc1) ** 2)
    qc1_var_loss = 0.5 * tf.reduce_mean(qc1_var + qc1_var_backup - 2 * tf.sqrt(qc1_var * qc1_var_backup))
    q1_loss = qr1_1_loss + qr2_1_loss + qc1_loss + qc1_var_loss

    qr1_2_loss = 0.5 * tf.reduce_mean((q2_backup - qr1_2) ** 2)
    qr2_2_loss = 0.5 * tf.reduce_mean((q2_backup - qr2_2) ** 2)
    qc2_loss = 0.5 * tf.reduce_mean((qc2_backup - qc2) ** 2)
    qc2_var_loss = 0.5 * tf.reduce_mean(qc2_var + qc2_var_backup - 2 * tf.sqrt(qc2_var * qc2_var_backup))
    q2_loss = qr1_2_loss + qr2_2_loss + qc2_loss + qc2_var_loss

    q_loss = q1_loss + q2_loss

    # 熵约束（按动作维度缩放）
    entropy_constraint_val1 = entropy_constraint * act_dim1
    entropy_constraint_val2 = entropy_constraint * act_dim2
    
    pi1_entropy = -tf.reduce_mean(logp_pi1)
    pi2_entropy = -tf.reduce_mean(logp_pi2)
    
    # 最小熵限制（防止熵坍缩）
    min_entropy1 = min_entropy_ratio * entropy_constraint_val1
    min_entropy2 = min_entropy_ratio * entropy_constraint_val2
    
    entropy_penalty1 = tf.maximum(0.0, min_entropy1 - pi1_entropy)
    entropy_penalty2 = tf.maximum(0.0, min_entropy2 - pi2_entropy)
    
    alpha1_loss = -alpha1 * (entropy_constraint_val1 - pi1_entropy) + 10.0 * entropy_penalty1
    alpha2_loss = -alpha2 * (entropy_constraint_val2 - pi2_entropy) + 10.0 * entropy_penalty2
    
    print('using entropy constraint: agent1=%.2f (min=%.2f), agent2=%.2f (min=%.2f)' % 
          (entropy_constraint_val1, min_entropy1, entropy_constraint_val2, min_entropy2))

    if use_costs:
        print('using cost constraint', cost_constraint_val)
        beta1_loss = beta1 * (cost_constraint_val - qc1 - pdf_cdf * tf.sqrt(qc1_var))
        beta2_loss = beta2 * (cost_constraint_val - qc2 - pdf_cdf * tf.sqrt(qc2_var))

    # ========== 优化器（动态学习率 + 梯度裁剪） ==========
    lr_actor_ph = tf.placeholder(tf.float32, shape=[], name='lr_actor')
    lr_critic_ph = tf.placeholder(tf.float32, shape=[], name='lr_critic')
    lr_context_ph = tf.placeholder(tf.float32, shape=[], name='lr_context')
    grad_clip_ph = tf.placeholder(tf.float32, shape=[], name='grad_clip')
    
    # 上下文编码器优化
    context_optimizer = MpiAdamOptimizer(learning_rate=lr_context_ph)
    context_grads_and_vars = context_optimizer.compute_gradients(
        context_loss, var_list=get_vars('context_encoder1') + get_vars('context_encoder2'))
    context_grads_clipped = [(tf.clip_by_norm(g, grad_clip_ph), v) for g, v in context_grads_and_vars if g is not None]
    train_context_op = context_optimizer.apply_gradients(context_grads_clipped)

    # Actor优化（智能体2学习率为智能体1的2倍，因其动作空间更小）
    with tf.control_dependencies([train_context_op]):
        pi1_optimizer = MpiAdamOptimizer(learning_rate=lr_actor_ph)
        pi1_grads_and_vars = pi1_optimizer.compute_gradients(pi1_loss, var_list=get_vars('main/pi1'))
        pi1_grads_clipped = [(tf.clip_by_norm(g, grad_clip_ph), v) for g, v in pi1_grads_and_vars if g is not None]
        train_pi1_op = pi1_optimizer.apply_gradients(pi1_grads_clipped)
        
        pi2_optimizer = MpiAdamOptimizer(learning_rate=lr_actor_ph * 2)
        pi2_grads_and_vars = pi2_optimizer.compute_gradients(pi2_loss, var_list=get_vars('main/pi2'))
        pi2_grads_clipped = [(tf.clip_by_norm(g, grad_clip_ph), v) for g, v in pi2_grads_and_vars if g is not None]
        train_pi2_op = pi2_optimizer.apply_gradients(pi2_grads_clipped)

    # Critic优化
    with tf.control_dependencies([train_pi1_op, train_pi2_op]):
        q_optimizer = MpiAdamOptimizer(learning_rate=lr_critic_ph)
        q_grads_and_vars = q_optimizer.compute_gradients(q_loss, var_list=get_vars('main/q'))
        q_grads_clipped = [(tf.clip_by_norm(g, grad_clip_ph), v) for g, v in q_grads_and_vars if g is not None]
        train_q_op = q_optimizer.apply_gradients(q_grads_clipped)

    # 熵系数优化
    if fixed_entropy_bonus is None:
        entreg_optimizer = MpiAdamOptimizer(learning_rate=lr)
        with tf.control_dependencies([train_q_op]):
            train_entreg_op = entreg_optimizer.minimize(alpha1_loss + alpha2_loss, var_list=get_vars('entreg'))

    # 成本惩罚系数优化
    if use_costs and fixed_cost_penalty is None:
        costpen_optimizer = MpiAdamOptimizer(learning_rate=lr * lr_scale)
        if fixed_entropy_bonus is None:
            with tf.control_dependencies([train_entreg_op]):
                train_costpen_op = costpen_optimizer.minimize(
                    tf.reduce_mean(beta1_loss) + tf.reduce_mean(beta2_loss),
                    var_list=get_vars('costpen'))
        else:
            with tf.control_dependencies([train_q_op]):
                train_costpen_op = costpen_optimizer.minimize(
                    tf.reduce_mean(beta1_loss) + tf.reduce_mean(beta2_loss),
                    var_list=get_vars('costpen'))

    # 目标网络软更新
    polyak_ph = tf.placeholder(tf.float32, shape=[], name='polyak')
    target_update = get_target_update('main', 'target', polyak_ph)

    # TD3风格延迟Actor更新：分离Critic-only和完整更新操作
    with tf.control_dependencies([train_context_op]):
        with tf.control_dependencies([train_q_op]):
            critic_only_update = tf.group([target_update])
    
    with tf.control_dependencies([train_context_op, train_pi1_op, train_pi2_op]):
        with tf.control_dependencies([train_q_op]):
            full_update = tf.group([target_update])

    if fixed_entropy_bonus is None:
        full_update = tf.group([full_update, train_entreg_op])
    if use_costs and fixed_cost_penalty is None:
        full_update = tf.group([full_update, train_costpen_op])
    
    grouped_update = full_update

    # ========== 动作获取（CTDE分布式执行：仅使用局部上下文） ==========
    def get_action1(o1, context1, deterministic=False):
        act_op = mu1 if deterministic else pi1
        return sess.run(act_op, feed_dict={
            o1_ph: o1.reshape(1, -1),
            context1_ph: context1.reshape(1, context_len, context_input_dim1)
        })[0]

    def get_action2(o2, context2, deterministic=False):
        act_op = mu2 if deterministic else pi2
        return sess.run(act_op, feed_dict={
            o2_ph: o2.reshape(1, -1),
            context2_ph: context2.reshape(1, context_len, context_input_dim2)
        })[0]

    def get_context_embedding(context1, context2):
        return sess.run([z1, z2, z_global], feed_dict={
            context1_ph: context1.reshape(1, context_len, context_input_dim1),
            context2_ph: context2.reshape(1, context_len, context_input_dim2)
        })

    def test_agent(num_wind=0, deterministic=True):
        o1, o2 = test_env.reset(n=num_wind, Train=Train)
        ep_ret1, ep_ret2, ep_cost1, ep_cost2, ep_len = 0, 0, 0, 0, 0
        u_g = np.zeros((n_gen, n_T))
        
        test_context_buffer1 = LocalContextBuffer(context_len, obs_dim1, act_dim1)
        test_context_buffer2 = LocalContextBuffer(context_len, obs_dim2, act_dim2)

        for i in range(max_ep_len):
            test_context1 = test_context_buffer1.get_context_for_decision()
            test_context2 = test_context_buffer2.get_context_for_decision()
            
            a1 = get_action1(o1, test_context1, deterministic)
            a2 = get_action2(o2, test_context2, deterministic)

            o1_next, r1, c1, o2_next, r2, c2, s_next, done = test_env.step(
                a1, a2, o1, o2, ep_len, Train=Train)

            test_context_buffer1.add_transition(o1, a1, r1, c1)
            test_context_buffer2.add_transition(o2, a2, r2, c2)

            ep_ret1 += r1
            ep_ret2 += r2
            ep_cost1 += c1
            ep_cost2 += c2
            ep_len += 1

            u_state = np.concatenate([
                (a1 > 0).astype(int),
                (a2 > 0).astype(int)
            ])
            u_g[:, i] = u_state

            o1, o2 = o1_next, o2_next

            if done:
                break

        return ep_ret1, ep_ret2, ep_cost1, ep_cost2, ep_len, u_g

    # ========== 会话配置 ==========
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)

    # ========== 测试模式 ==========
    if not Train:
        saver = tf.train.Saver()
        try:
            saver.restore(sess, tf.train.latest_checkpoint('./saved_models/CMA_BC_RSAC_%d' % args.Model_version))

            num_wind = 0
            start = time.time()
            ep_ret1, ep_ret2, ep_cost1, ep_cost2, ep_len, u_g = test_agent(num_wind=num_wind)
            end = time.time()
            print("求解时间:%.4f秒" % (end - start))

            print('\n===================CMA-RSAC Test Result====================')
            print('  The scenario of wind   |   %d' % num_wind)
            print('  Agent1 Return          |   %.2f' % ep_ret1)
            print('  Agent2 Return          |   %.2f' % ep_ret2)
            print('  Agent1 Cost            |   %.2f' % ep_cost1)
            print('  Agent2 Cost            |   %.2f' % ep_cost2)
            print('  Total Return           |   %.2f' % (ep_ret1 + ep_ret2))
            print('  Total Cost             |   %.2f' % (ep_cost1 + ep_cost2))
            print('============================================================')

            sns.heatmap(u_g, linewidths=0.05, linecolor='red', annot=False)
            plt.title('CMA-RSAC Unit Commitment Result')
            plt.xlabel('Time Period')
            plt.ylabel('Generator')

            if args.SaveFig:
                now = datetime.datetime.now()
                savefname = './Debug_record/%d.%d.%d/Figture' % (now.year, now.month, now.day)
                os.makedirs(savefname, exist_ok=True)
                plt.savefig(savefname + '/CMA_BC_RSAC_Output_%d_%d.png' % (num_wind, args.Model_version), dpi=500)

            plt.show()

        except Exception as e:
            print(f'无法找到模型路径或模型网络不对应: {e}')

    # ========== 训练模式 ==========
    else:
        writer = tf.summary.FileWriter(".logs/", sess.graph)

        if Continue_Training:
            if Model_Portion:
                target_init = get_target_update('main', 'target', 0.0)
                sess.run(tf.global_variables_initializer())
                sess.run(target_init)
                sess.run(sync_all_params())
                Loading_model = [get_vars('main/q'), get_vars('target'), get_vars('costpen'), get_vars('entreg')]
                saver = tf.train.Saver(Loading_model)
                saver.restore(sess, tf.train.latest_checkpoint('./saved_models/CMA_BC_RSAC_%d' % args.Model_version))
            else:
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint('./saved_models/CMA_BC_RSAC_%d' % args.Model_version))
        else:
            target_init = get_target_update('main', 'target', 0.0)
            sess.run(tf.global_variables_initializer())
            sess.run(target_init)
            sess.run(sync_all_params())

        logger.setup_tf_saver(sess, inputs={'o1': o1_ph, 'o2': o2_ph, 'a1': a1_ph, 'a2': a2_ph, 
                                           'context1': context1_ph, 'context2': context2_ph},
                              outputs={'mu1': mu1, 'pi1': pi1, 'mu2': mu2, 'pi2': pi2, 
                                      'z1': z1, 'z2': z2})

        start_time = time.time()

        o1, o2 = env.reset()
        s = build_global_state(o1, o2)

        ep_ret1, ep_ret2, ep_cost1, ep_cost2, ep_len = 0, 0, 0, 0, 0
        total_steps = steps_per_epoch * epochs * max_ep_len

        vars_to_get = dict(
            LossPi1=pi1_loss, LossPi2=pi2_loss,
            LossQR1_1=qr1_1_loss, LossQR1_2=qr1_2_loss,
            LossQC1=qc1_loss, LossQC2=qc2_loss,
            LossQC1Var=qc1_var_loss, LossQC2Var=qc2_var_loss,
            QR1_1Vals=qr1_1, QR1_2Vals=qr1_2,
            QC1Vals=qc1, QC2Vals=qc2,
            LogPi1=logp_pi1, LogPi2=logp_pi2,
            Pi1Entropy=pi1_entropy, Pi2Entropy=pi2_entropy,
            Alpha1=alpha1, Alpha2=alpha2,
            LogAlpha1=log_alpha1, LogAlpha2=log_alpha2,
            LossAlpha1=alpha1_loss, LossAlpha2=alpha2_loss,
            ContextLoss=context_loss,
            KLLoss=kl_loss,
            ReconLoss=recon_loss
        )
        if use_costs:
            vars_to_get.update(dict(
                Beta1=beta1, Beta2=beta2,
                LogBeta1=log_beta1, LogBeta2=log_beta2,
                LossBeta1=beta1_loss, LossBeta2=beta2_loss
            ))

        print('Starting CMA-RSAC training', proc_id())

        # 初始化参数调度器
        if use_scheduler:
            scheduler = TrainingScheduler(total_epochs=epochs)
            print('\n========== 启用自适应参数调度 ==========')
            print('训练阶段规划:')
            print('  阶段1 (0-2000 epoch): 探索阶段')
            print('  阶段2 (2000-6000 epoch): 稳定学习阶段')
            print('  阶段3 (6000-12000 epoch): 精细调整阶段')
            print('  阶段4 (12000-16000 epoch): 收敛阶段')
            print('============================================\n')
            sched_params = scheduler.get_params(0)
            current_lr_actor = sched_params['lr_actor']
            current_lr_critic = sched_params['lr_critic']
            current_lr_context = sched_params['lr_context']
            current_grad_clip = sched_params['grad_clip']
            current_polyak = sched_params['polyak']
            current_actor_update_freq = sched_params['actor_update_freq']
        else:
            current_lr_actor = lr_actor
            current_lr_critic = lr_critic
            current_lr_context = lr
            current_grad_clip = grad_clip
            current_polyak = polyak
            current_actor_update_freq = actor_update_freq

        # 主训练循环
        cum_cost1, cum_cost2 = 0, 0
        local_steps = 0
        local_steps_per_epoch = steps_per_epoch // num_procs()
        local_batch_size = batch_size // num_procs()
        epoch_start_time = time.time()

        for t in range(total_steps // num_procs()):
            # 步骤1: 获取决策前的上下文（历史t-H到t-1，严格时间因果性）
            context1, context2 = replay_buffer.get_context_for_decision()
            
            # 步骤2: 选择动作（分布式执行，仅使用局部上下文）
            if t > local_start_steps:
                a1 = get_action1(o1, context1)
                a2 = get_action2(o2, context2)
            else:
                a1, a2 = env.action_sample()

            # 步骤3: 环境交互
            o1_next, r1, c1, o2_next, r2, c2, s_next, done = env.step(a1, a2, o1, o2, ep_len)

            r1 *= reward_scale
            r2 *= reward_scale
            ep_ret1 += r1
            ep_ret2 += r2
            ep_cost1 += c1
            ep_cost2 += c2
            ep_len += 1
            local_steps += 1

            cum_cost1 += c1
            cum_cost2 += c2

            # 步骤4: 存储经验（使用决策前的上下文，不泄露当前步信息）
            replay_buffer.store(o1, o2, a1, a2, r1, r2, o1_next, o2_next, 
                               s, s_next, done, c1, c2, context1, context2)

            # 步骤5: 将当前步加入历史（用于下一步决策）
            replay_buffer.add_transition_to_history(o1, a1, r1, c1, o2, a2, r2, c2)

            o1, o2, s = o1_next, o2_next, s_next

            if done or (ep_len == max_ep_len):
                logger.store(EpRet1=ep_ret1, EpRet2=ep_ret2, EpCost1=ep_cost1, EpCost2=ep_cost2)
                o1, o2 = env.reset(n=np.random.randint(n_wind_data))
                s = build_global_state(o1, o2)
                replay_buffer.reset_context()
                ep_ret1, ep_ret2, ep_cost1, ep_cost2, ep_len = 0, 0, 0, 0, 0

            # 更新网络
            if t > 0 and t % update_freq == 0:
                for j in range(update_freq):
                    batch = replay_buffer.sample_batch(local_batch_size)
                    
                    feed_dict = {
                        o1_ph: batch['obs1'],
                        o1_next_ph: batch['obs1_next'],
                        a1_ph: batch['acts1'],
                        r1_ph: batch['rews1'],
                        c1_ph: batch['costs1'],
                        o2_ph: batch['obs2'],
                        o2_next_ph: batch['obs2_next'],
                        a2_ph: batch['acts2'],
                        r2_ph: batch['rews2'],
                        c2_ph: batch['costs2'],
                        s_ph: batch['global_state'],
                        s_next_ph: batch['global_state_next'],
                        context1_ph: batch['context1'],
                        context2_ph: batch['context2'],
                        d_ph: batch['done'],
                        lr_actor_ph: current_lr_actor,
                        lr_critic_ph: current_lr_critic,
                        lr_context_ph: current_lr_context,
                        grad_clip_ph: current_grad_clip,
                        polyak_ph: current_polyak,
                    }

                    if t < local_update_after:
                        logger.store(**sess.run(vars_to_get, feed_dict))
                    else:
                        # TD3延迟Actor更新
                        if j % current_actor_update_freq == 0:
                            values, _ = sess.run([vars_to_get, full_update], feed_dict)
                        else:
                            values, _ = sess.run([vars_to_get, critic_only_update], feed_dict)
                        logger.store(**values)

                ETA = (time.time() - epoch_start_time) * (total_steps // update_freq - t // update_freq)
                print('CMA-RSAC Training: [epoch:%d|%d] [batch:%d|%d] ETA %d:%d:%d' %
                      (t // (local_steps_per_epoch * max_ep_len), epochs - 1,
                       (t % (local_steps_per_epoch * max_ep_len) // max_ep_len), steps_per_epoch - 1,
                       ETA // 3600, ETA % 3600 // 60, ETA % 60))
                epoch_start_time = time.time()

            # Epoch结束
            if t > 0 and t % (local_steps_per_epoch * max_ep_len) == 0:
                epoch = t // (local_steps_per_epoch * max_ep_len)
                cumulative_cost = mpi_sum(cum_cost1 + cum_cost2)
                cost_rate = cumulative_cost / ((epoch + 1) * steps_per_epoch)

                # 更新调度器参数
                if use_scheduler:
                    sched_params = scheduler.get_params(epoch)
                    current_lr_actor = sched_params['lr_actor']
                    current_lr_critic = sched_params['lr_critic']
                    current_lr_context = sched_params['lr_context']
                    current_grad_clip = sched_params['grad_clip']
                    current_polyak = sched_params['polyak']
                    current_actor_update_freq = sched_params['actor_update_freq']

                # 保存模型
                if (epoch % save_freq == 0) or (epoch == epochs - 1):
                    saver = tf.train.Saver()
                    saver.save(sess, './saved_models/CMA_BC_RSAC_%d/Model' % epoch)
                    try:
                        now_time = datetime.datetime.now()
                        os.makedirs('./Debug_record/%d.%d.%d' % (now_time.year, now_time.month, now_time.day),
                                    exist_ok=True)
                        saver.save(sess, './Debug_record/%d.%d.%d/CMA_BC_RSAC_%d/Model' %
                                   (now_time.year, now_time.month, now_time.day, epoch))
                        Path_xls = './Debug_record/%d.%d.%d/ReplayBuffer' % (
                            now_time.year, now_time.month, now_time.day)
                        os.makedirs(Path_xls, exist_ok=True)
                        replay_buffer.savebuffer('./logger/MABuffer.xlsx')
                        replay_buffer.savebuffer(Path_xls=Path_xls + '/MABuffer_%d.xlsx' % now_time.hour)
                    except:
                        print('保存路径出错')

                # 记录日志
                logger.log_tabular('Epoch', epoch)
                
                metrics_with_minmax = [
                    'EpRet1', 'EpRet2', 'EpCost1', 'EpCost2',
                    'LossPi1', 'LossQR1_1', 'LossQC1', 'LossQC1Var', 'Pi1Entropy',
                    'LossPi2', 'LossQR1_2', 'LossQC2', 'LossQC2Var', 'Pi2Entropy',
                    'QR1_1Vals', 'QR1_2Vals', 'QC1Vals', 'QC2Vals'
                ]
                for metric in metrics_with_minmax:
                    logger.log_tabular(metric, with_min_and_max=True)
                
                logger.log_tabular('ContextLoss', average_only=True)
                logger.log_tabular('KLLoss', average_only=True)
                logger.log_tabular('ReconLoss', average_only=True)
                
                logger.log_tabular('CumulativeCost', cumulative_cost)
                logger.log_tabular('CostRate', cost_rate)
                logger.log_tabular('Alpha1', average_only=True)
                logger.log_tabular('Alpha2', average_only=True)
                if use_costs:
                    logger.log_tabular('Beta1', average_only=True)
                    logger.log_tabular('Beta2', average_only=True)
                logger.log_tabular('TotalTime', time.time() - start_time)
                
                if use_scheduler:
                    phase_info = scheduler.get_phase_info(epoch)
                    print(f'  当前训练阶段: {phase_info}')
                    print(f'  当前参数: lr_actor={current_lr_actor:.2e}, lr_critic={current_lr_critic:.2e}, '
                          f'actor_update_freq={current_actor_update_freq}, grad_clip={current_grad_clip:.2f}')
                
                logger.dump_tabular()

        writer.close()


if __name__ == '__main__':
    mpi_fork(args.cpu)

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger_kwargs = setup_logger_kwargs(args.exp_name)
    logger_kwargs = args.logger_kwargs_str

    cma_bc_rsac(lambda: unPower(uncertain=True, n_sce=20),
             ac_kwargs_actor=dict(hidden_sizes=args.hidden_sizes_actor, cnn_sizes=args.cnn_actor),
             ac_kwargs_critic=dict(hidden_sizes=args.hidden_sizes_critic, cnn_sizes=args.cnn_critic),
             ac_kwargs_var=dict(hidden_sizes=args.hidden_sizes_var, cnn_sizes=args.cnn_var),
             gamma=args.gamma, cl=args.cl, seed=args.seed, epochs=args.epochs,
             replay_size=24000, batch_size=args.batch_size,
             polyak=args.polyak,
             lr=args.lr, lr_actor=args.lr_actor, lr_critic=args.lr_critic,
             logger_kwargs=logger_kwargs, steps_per_epoch=args.steps_per_epoch,
             update_freq=args.update_freq, render=args.render,
             local_start_steps=args.local_start_steps, save_freq=args.save_freq,
             local_update_after=args.local_update_after,
             fixed_entropy_bonus=args.fixed_entropy_bonus, entropy_constraint=args.entropy_constraint,
             fixed_cost_penalty=args.fixed_cost_penalty, cost_constraint=args.cost_constraint,
             cost_lim=args.cost_lim,
             lr_scale=args.lr_s, damp_scale=args.damp_s, Train=args.Train,
             Continue_Training=args.Continue,
             Model_Portion=args.Portion, load_buffer=args.load_buffer,
             context_len=args.context_len, latent_dim=args.latent_dim,
             encoder_hidden=args.encoder_hidden, decoder_hidden=args.decoder_hidden,
             kl_weight=args.kl_weight, recon_weight=args.recon_weight,
             q_clip_max=args.q_clip_max, grad_clip=args.grad_clip,
             actor_update_freq=args.actor_update_freq,
             lr_decay_start=args.lr_decay_start,
             lr_decay_rate=args.lr_decay_rate,
             min_entropy_ratio=args.min_entropy_ratio,
             use_scheduler=args.use_scheduler
             )

    if args.Train:
        Path_txt = './logger/progress.txt'
        Path_xls = './logger/progress.xls'
        logger_Dict = save_data.txt_xls(Path_txt, Path_xls, convert=True)

        plot_configs = [
            ('EpRet1', 'Agent1 Episode Return', '#496C88', True),
            ('EpRet2', 'Agent2 Episode Return', '#82B0D2', True),
            ('EpCost1', 'Agent1 Episode Cost', '#FA7F6F', True),
            ('EpCost2', 'Agent2 Episode Cost', '#FFBE7A', True),
            ('CostRate', 'Cost Rate', None, False),
            ('LossPi1', 'Agent1 Actor Loss', '#8ECFC9', True),
            ('LossPi2', 'Agent2 Actor Loss', '#BEB8DC', True),
            ('Pi1Entropy', 'Agent1 Policy Entropy', '#FE4365', True),
            ('Pi2Entropy', 'Agent2 Policy Entropy', '#F38181', True),
            ('LossQR1_1', 'Agent1 Q-Reward Loss', '#54478C', True),
            ('LossQC1', 'Agent1 Q-Cost Loss', '#E6B33D', True),
            ('LossQC1Var', 'Agent1 Q-Cost Var Loss', '#F29E4C', True),
            ('LossQR1_2', 'Agent2 Q-Reward Loss', '#2C699A', True),
            ('LossQC2', 'Agent2 Q-Cost Loss', '#048A81', True),
            ('LossQC2Var', 'Agent2 Q-Cost Var Loss', '#83E377', True),
            ('QR1_1Vals', 'Agent1 Q-Reward Values', '#0DB39E', True),
            ('QR1_2Vals', 'Agent2 Q-Reward Values', '#16DB93', True),
            ('QC1Vals', 'Agent1 Q-Cost Values', '#B9E769', True),
            ('QC2Vals', 'Agent2 Q-Cost Values', '#EFEA5A', True),
            ('ContextLoss', 'Context Encoder Total Loss', '#9B59B6', False),
            ('KLLoss', 'Context KL Divergence Loss', '#3498DB', False),
            ('ReconLoss', 'Context Reconstruction Loss', '#E74C3C', False),
        ]
        
        for item, title, color, tsplot in plot_configs:
            try:
                Fig(logger_Dict, item, Title=title, c=color, SaveFig=args.SaveFig, tsplot=tsplot)
            except Exception as e:
                print(f'绘图 {item} 失败: {e}')

        if args.SaveFig:
            now_time = datetime.datetime.now()
            des = f'./Debug_record/{now_time.year}.{now_time.month}.{now_time.day}/logger_{now_time.hour}'
            os.makedirs(des, exist_ok=True)
            
            src = './logger'
            for file in os.listdir(src):
                full_file_name = os.path.join(src, file)
                if os.path.isfile(full_file_name):
                    try:
                        shutil.copy(full_file_name, des)
                    except Exception as e:
                        print(f'保存路径出错: {e}')

            with open(f'{des}/args.txt', "w", encoding="utf-8") as log_args:
                for key, value in vars(args).items():
                    log_args.write(f'{key}: {value}\n')