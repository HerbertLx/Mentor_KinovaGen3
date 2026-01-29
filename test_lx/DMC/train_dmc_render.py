import os
import gymnasium as gym
import numpy as np
from dm_control import suite
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback

# 1. 环境配置
os.environ['MUJOCO_GL'] = 'egl'  # 无头模式渲染

class WandbVideoCallback(BaseCallback):
    """自定义回调：每隔一段时间渲染一段视频并上传到 WandB"""
    def __init__(self, eval_env, render_freq=10000):
        super().__init__()
        self.eval_env = eval_env
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            # 简单录制一个片段
            screens = []
            obs, _ = self.eval_env.reset()
            for _ in range(200):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                # 渲染视觉图
                screens.append(self.eval_env.render().transpose(2, 0, 1))
                if terminated or truncated: break
            
            # 上传到 WandB
            wandb.log({"video": wandb.Video(np.array(screens), fps=30, format="mp4")})
        return True

def make_dmc_env():
    # 使用 shimmy 库将 DMC 转换为 Gymnasium 接口
    env = suite.load(domain_name="dog", task_name="walk")
    env = DmControlCompatibilityV0(env, render_mode="rgb_array")
    env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
    return env

if __name__ == "__main__":
    # 2. 初始化 WandB
    run = wandb.init(
        project="dog_walk_study",
        config={"algorithm": "SAC", "total_timesteps": 1000000},
        sync_tensorboard=True, 
        monitor_gym=True,
        save_code=True,
    )

    # 3. 创建环境
    env = make_dmc_env()

    # 4. 定义模型 (SAC 是处理连续动作空间最成熟的模型之一)
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

    # 5. 开始训练并记录视频
    model.learn(
        total_timesteps=run.config["total_timesteps"],
        callback=[
            WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
            WandbVideoCallback(make_dmc_env(), render_freq=5000)
        ]
    )

    run.finish()