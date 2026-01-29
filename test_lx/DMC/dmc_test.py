import os
import cv2
import numpy as np
import torch
from dm_control import suite
# 强制使用 glfw
os.environ['MUJOCO_GL'] = 'glfw' 

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

# 回调函数：用于实时打印训练指标
class InfoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(InfoCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            # 从 SB3 内部记录器提取数据
            logs = self.model.logger.name_to_value
            actor_loss = logs.get("train/actor_loss", 0)
            critic_loss = logs.get("train/loss", 0) 
            ent_coeff = logs.get("train/ent_coeff", 0)
            
            print(f"Step: {self.num_timesteps} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f} | Alpha: {ent_coeff:.4f}")
        return True

class Workspace:
    def __init__(self):
        self.domain_name = "dog"
        self.task_name = "walk"
        
        # 强制检查 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- 正在使用设备: {self.device} ---")

        # 保存原生环境，以便后续直接操作底层 physics
        self.raw_env = suite.load(domain_name=self.domain_name, task_name=self.task_name)
        
        # 包装环境
        self.env = DmControlCompatibilityV0(self.raw_env, render_mode="rgb_array")
        self.env = gym.wrappers.RescaleAction(self.env, min_action=-1, max_action=1)
        
        # 38 维动作空间，使用三层 512 的深度网络
        policy_kwargs = dict(net_arch=[512, 512, 512]) 
        
        self.agent = SAC(
            "MultiInputPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=0, 
            device=self.device
        )

        self.global_step = 0
        self.info_callback = InfoCallback()
        self.info_callback.init_callback(self.agent)

    def render_live(self):
        """
        修正后的渲染函数：通过物理引擎直接渲染大窗口
        """
        # 直接调用物理引擎渲染，不受 Gymnasium 接口限制
        # height/width 设为你想要的大小
        # camera_id=0 为全局视角，camera_id=1 通常是追踪视角
        frame = self.raw_env.physics.render(height=720, width=1280, camera_id=0)
        
        # 转换 BGR 用于 OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 画面叠加信息
        cv2.putText(frame, f"Step: {self.global_step}", (40, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # 显示窗口
        cv2.imshow("Dog Walk - GPU Training Live", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    def train(self, total_steps=1000000):
        print("开始训练...")
        obs, _ = self.env.reset()
        episode_reward = 0
        
        for _ in range(total_steps):
            action, _ = self.agent.predict(obs, deterministic=False)
            new_obs, reward, terminated, truncated, info = self.env.step(action)
            episode_reward += reward
            
            # 学习一步
            self.agent.learn(total_timesteps=1, reset_num_timesteps=False, callback=self.info_callback)
            
            # 控制直播频率：每 500 步循环中，显示最后 100 步
            if self.global_step % 500 > 400:
                self.render_live()
            
            obs = new_obs
            self.global_step += 1
            
            # 当回合结束或强制 500 步重置（增加观察样本）
            if terminated or truncated or (self.global_step % 500 == 0):
                print(f"--- Episode End | Reward: {episode_reward:.2f} | Total Steps: {self.global_step} ---")
                obs, _ = self.env.reset()
                episode_reward = 0

        cv2.destroyAllWindows()

if __name__ == "__main__":
    workspace = Workspace()
    workspace.train()

"""
DMC Dog Walk 任务观测空间 (Observation Space) 详解：
--------------------------------------------------
1. joint_angles (73): 
   - 包含机器人所有关节的广义坐标（角度），用于描述全身姿态。
2. joint_velocities (73): 
   - 关节角速度，反映动作的瞬时动量和运动趋势。
3. torso_pelvis_height (2): 
   - 躯干和盆骨离地高度。是判断平衡状态（是否跌倒）的关键指标。
4. z_projection (9): 
   - 躯干各轴在世界坐标系Z轴上的投影，用于感知重力方向及身体倾斜度。
5. torso_com_velocity (3): 
   - 躯干质心(Center of Mass)的3D线速度。AI 学习前进动力的核心参考。
6. inertial_sensors (9): 
   - 模拟 IMU 数据（加速度计/陀螺仪），提供类似生物“内耳”的平衡反馈。
7. foot_forces (12): 
   - 四足末端受到的地面反作用力矢量，涉及复杂的“富接触”动力学。
8. touch_sensors (4): 
   - 二值信号，指示每只足端是否与地面发生物理接触。
9. actuator_state (38): 
   - 38个执行器的当前输出状态。对应 Dog 模型的高维动作控制维度。
--------------------------------------------------
总观测维度非常大，SAC 需要通过这些特征学习高度协同的肢体控制策略。
"""