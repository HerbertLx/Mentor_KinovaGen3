import warnings

# 过滤掉所有过期的警告，保持终端输出整洁
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

# 强制设置 MKL 使用 Intel 核心优化（解决某些环境下 MKL 的性能问题）
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# 设置 MuJoCo 使用 EGL 渲染（适用于没有显示器的服务器/无头模式渲染）
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import utils
import torch
from dm_env import specs

import dmc

from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb
import re

from utils import models_tuple
from copy import deepcopy

# 开启 cuDNN 自动优化，会自动寻找最适合当前硬件的卷积实现算法，加速训练
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    """
    实例化智能体（Agent）的工厂函数。
    
    输入参数：
        obs_spec: 环境的观测空间规格（包含维度信息）。
        action_spec: 环境的动作空间规格（包含维度和范围）。
        cfg: Hydra 的配置对象，包含智能体的参数（如隐藏层维度、学习率等）。
    
    内部逻辑：
        1. 将环境的规格信息写入配置对象，确保智能体网络输入输出层维度正确。
        2. 使用 hydra.utils.instantiate 根据配置文件动态创建智能体实例。
        
    输出：
        返回一个初始化的 Agent 对象。
    """
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    """
    训练工作流管理类（核心逻辑封装）。
    
    主要职责：
        1. 环境初始化、Logger（日志）、WandB 的配置。
        2. 管理训练循环（Training Loop）和评估循环（Evaluation Loop）。
        3. 处理数据的存储、读取以及模型的保存/加载。
    """
    def __init__(self, cfg):
        """
        初始化工作空间。
        输入：cfg - 完整的 Hydra 配置。
        """
        self.work_dir = Path.cwd() # 获取当前 Hydra 自动生成的实验目录
        self.cfg = cfg
        print("#"*20)
        print(f'\nworkspace: {self.work_dir}')
        for key in self.cfg:

            if key == 'agent':
                for agent_key in self.cfg['agent']:
                    if agent_key == 'obs_shape' or agent_key == 'action_shape':
                        print(f"self.cfg['agent']['{agent_key}'] = ")
                    else:
                        print(f"self.cfg['agent']['{agent_key}'] = {self.cfg['agent'][agent_key]}")
            else:
                print(f"self.cfg['{key}'] = {self.cfg[key]}")


        print()
        
        self.last_save_step = -9999 # 用于记录上一次保存 Snapshot 的步数
        
        # --- WandB 初始化 ---
        if self.cfg.use_wandb:
            # 根据任务名和随机种子生成实验名称
            exp_name = '_'.join([cfg.task_name, str(cfg.seed)])
            # 从 agent 目标路径中正则匹配出算法名称（如 mentor）
            group_name = re.search(r'\.(.+)\.', cfg.agent._target_).group(1)
            name_1 = cfg.task_name
            name_2 = group_name
            try:
                name_2 += '_' + cfg.title
            except:
                pass
            name_3 = exp_name
            # 初始化 WandB 项目
            wandb.init(project=name_1,
                       group=name_2,
                       name=name_3,
                       config=cfg)
        
        # 设置全局随机种子
        utils.set_seed_everywhere(cfg.seed)
        # 设置计算设备（CUDA 或 CPU）
        self.device = torch.device(cfg.device)
        self._discount = cfg.discount
        self._nstep = cfg.nstep
        
        # 执行子系统设置（环境、Buffer 等）
        self.setup()
        
        # 创建核心 Agent
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), self.cfg.agent)
        
        # 计时器
        self.timer = utils.Timer()
        self._global_step = 0    # 记录决策步数
        self._global_episode = 0 # 记录完成的回合数

    def setup(self):
        """
        内部辅助函数：初始化日志、环境、经验回放池和视频录制器。
        """
        # 创建 Logger 对象，支持 Tensorboard、WandB 和 CSV 存储
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)
        
        # 创建训练环境和评估环境
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)
        
        # 定义 Replay Buffer 存储的数据规格（状态、动作、奖励、折扣系数）
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1, ), np.float32, 'reward'),
                      specs.Array((1, ), np.float32, 'discount'))

        # 实例化存储器
        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        
        # 创建数据加载器（Loader）和经验池缓存
        self.replay_loader, self.buffer = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers, self.cfg.save_snapshot,
            self._nstep,
            self._discount)
        
        self._replay_iter = None

        # 视频录制器初始化
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)

    @property
    def global_step(self):
        """当前决策步数（不含 action repeat）"""
        return self._global_step

    @property
    def global_episode(self):
        """当前总回合数"""
        return self._global_episode

    @property
    def global_frame(self):
        """当前总帧数（决策步数 * 动作重复次数）"""
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        """延迟初始化并返回 Replay Buffer 的迭代器"""
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        """
        运行评估流程。
        主体逻辑：
            1. 切换至评估环境。
            2. 使用 Agent 的评估模式（通常是确定性动作）。
            3. 录制视频并统计平均奖励。
        """
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            # 仅录制第一场评估回合的视频以节省开销
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            # 将视频保存为 mp4
            self.video_recorder.save(f'{self.global_frame}.mp4')
            
        # 记录评估统计数据
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        """
        核心训练循环。
        主体逻辑：
            1. 循环直到达到预设的总帧数。
            2. 收集环境轨迹存入 Replay Buffer。
            3. 在达到一定步数后，定期更新（Update）智能体。
            4. 定期调用 eval() 进行测试。
        """
        # 定义各类周期性触发的谓词函数
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step) # 存入初始状态
        metrics = None
        print("start training")
        
        while train_until_step(self.global_step):
            # --- 回合结束后的处理 ---
            if time_step.last():
                self._global_episode += 1
                
                # 如果有训练指标，将其记录到日志中
                if metrics is not None:
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                
                # 特殊逻辑：如果 Agent 有 tp_set 属性（通常是 Mentor 算法特有的），更新其优先级队列
                if hasattr(self.agent, 'tp_set'):
                    self.agent.tp_set.add(episode_reward,\
                                            deepcopy(self.agent.actor),\
                                            deepcopy(self.agent.critic),\
                                            deepcopy(self.agent.critic_target),\
                                            deepcopy(self.agent.value_predictor),\
                                            moe=deepcopy(self.agent.actor.moe.experts),\
                                            gate=deepcopy(self.agent.actor.moe.gate))                    
                
                # 环境重置
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                
                # 检查是否需要保存模型快照
                if self.cfg.save_snapshot and self.global_step - self.last_save_step >= self.cfg.save_interval:
                    self.last_save_step = self.global_step
                    self.save_snapshot(self.global_step)
                    
                episode_step = 0
                episode_reward = 0

            # --- 定期评估 ---
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # --- 动作选择 ---
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # --- 智能体参数更新 ---
            # 只有当初始“种子帧”收集完成后才开始训练更新
            if not seed_until_step(self.global_step):
                # 每隔 update_every_steps 步进行一次反向传播更新
                metrics = self.agent.update(
                    self.replay_iter, self.global_step
                ) if self.global_step % self.cfg.update_every_steps == 0 else dict()
                
                # 记录算法内部特有的指标
                if hasattr(self.agent, 'tp_set'):
                    metrics = self.agent.tp_set.log(metrics)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # --- 环境推进 ---
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self, step_id=None):
        """
        保存当前训练状态的快照。
        输入：step_id - 当前步数（用于区分不同的备份文件）。
        """
        if step_id is None:
            snapshot = self.work_dir / 'snapshot.pt'
        else:
            if not os.path.exists(str(self.work_dir) + '/snapshots'):
                os.makedirs(str(self.work_dir) + '/snapshots')
            snapshot = self.work_dir / 'snapshots' / 'snapshot_{}.pt'.format(step_id)
            
        # 选择需要序列化的属性
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, step_id=None):
        """
        加载之前保存的模型快照以恢复训练。
        输入：step_id - 想要加载的具体步数 ID。
        """
        if step_id is None:
            snapshot = self.work_dir / 'snapshot.pt'
        else:
            snapshot = self.work_dir / 'snapshots' / 'snapshot_{}.pt'.format(step_id)
        if not snapshot.exists():
            raise FileNotFoundError(f"Snapshot {snapshot} not found.")
        
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        # 将保存的状态恢复到当前对象中
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config_fast')
def main(cfgs):
    """
    程序入口点（Hydra 装饰器管理配置）。
    """
    from train_dmc import Workspace as W
    root_dir = Path.cwd()
    
    # 1. 初始化工作空间
    workspace = W(cfgs)
    
    # 2. 确定是否需要断点续训
    if cfgs.load_from_id:
        # 从特定 ID 的快照加载
        snapshot = root_dir / 'snapshots' / f'snapshot_{cfgs.load_id}.pt'
    else:
        # 从最新的快照加载
        snapshot = root_dir / 'snapshot.pt'
        
    # 如果文件存在，执行恢复逻辑
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()        
    
    # 3. 启动主训练程序
    workspace.train()


if __name__ == '__main__':
    main()