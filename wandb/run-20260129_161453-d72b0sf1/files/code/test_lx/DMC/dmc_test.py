import os
import numpy as np
import mujoco
import dm_control.suite as suite
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces

# --- 1. 将 DMC 包装为 Gymnasium 兼容的标准环境 ---
class DMControlWrapper(gym.Env):
    def __init__(self, domain_name, task_name, render_mode="rgb_array"):
        super().__init__()
        self.env = suite.load(domain_name, task_name)
        self.render_mode = render_mode
        
        # 提取动作空间 (Dog Walk 是 38 维)
        spec = self.env.action_spec()
        self.action_space = spaces.Box(low=spec.minimum, high=spec.maximum, dtype=np.float32)
        
        # 简单处理：将字典观测值拼接成一个长向量
        obs_spec = self.env.observation_spec()
        total_dim = sum([np.prod(v.shape) for v in obs_spec.values()])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

    def _get_obs(self, time_step):
        # 拼接字典中的所有观测项
        return np.concatenate([v.flatten() for v in time_step.observation.values()])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        time_step = self.env.reset()
        return self._get_obs(time_step), {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._get_obs(time_step)
        reward = time_step.reward or 0.0
        terminated = time_step.last()
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self):
        # 渲染 Dog 的画面
        return self.env.physics.render(height=240, width=320, camera_id=0)

# --- 2. 训练与 WandB 可视化 ---
def train_dog():
    # 初始化 WandB
    run = wandb.init(
        project="dog-walk-baseline",
        name="sac_dog_walk_v1",
        sync_tensorboard=True,  # 自动同步 SB3 的日志
        monitor_gym=True,       # 自动记录训练视频
        save_code=True,
    )

    # 创建环境
    env = DMControlWrapper("dog", "walk")

    # 定义模型 (SAC)
    # Dog 任务动作维度高达 38，建议增大网络规模
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=f"runs/{run.id}",
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
    )

    # 自定义实时渲染与上传 WandB 的逻辑 (每 5000 步上传一张图片)
    class VideoLogCallback(WandbCallback):
        def _on_step(self) -> bool:
            if self.n_calls % 5000 == 0:
                img = self.training_env.render()
                wandb.log({"live_render": wandb.Image(img)}, step=self.num_timesteps)
            return True

    # 开始训练
    print("开始训练 Dog Walk 任务...")
    model.learn(
        total_timesteps=500000, 
        callback=VideoLogCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
    )

    # 保存最终模型
    model.save("sac_dog_walk_final")
    run.finish()

if __name__ == "__main__":
    # 如果在无显示器服务器运行，请确保设置 EGL
    os.environ['MUJOCO_GL'] = 'egl'
    train_dog()