import os
import cv2
import numpy as np
import torch
from dm_control import suite
# 强制使用 egl 可能会导致本地弹窗失败，如果在有显示器的 Ubuntu 上，可以注释掉这行或改为 'glfw'
os.environ['MUJOCO_GL'] = 'glfw' 

from stable_baselines3 import SAC
import gymnasium as gym
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

class Workspace:
    def __init__(self):
        # 1. 环境初始化 (仿 MENTOR 风格)
        self.domain_name = "dog"
        self.task_name = "walk"
        
        # 加载原生的 DMC 环境
        raw_env = suite.load(domain_name=self.domain_name, task_name=self.task_name)
        
        # 包装成 Gymnasium 接口，方便 SAC 使用
        self.env = DmControlCompatibilityV0(raw_env, render_mode="rgb_array")
        self.env = gym.wrappers.RescaleAction(self.env, min_action=-1, max_action=1)
        
        # 2. 算法初始化
        # Dog 任务动作空间 38 维，属于极高维度，SAC 需要较大的网络
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
            verbose=1
        )

        self.global_step = 0
        print(f"成功加载任务: {self.domain_name}_{self.task_name}")
        print(f"动作空间: {self.env.action_space}")

    def render_live(self, obs):
        """实时渲染直播函数"""
        # 获取环境渲染图
        frame = self.env.render() # 获取的是 RGB 阵列
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # OpenCV 需要 BGR
        
        # 在画面上打上当前步数的标签
        cv2.putText(frame, f"Step: {self.global_step}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Dog Walk Live Training", frame)
        # 等待 1ms 刷新窗口，按 'q' 可以退出渲染（不停止训练）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

    def train(self, total_steps=1000000):
        print("开始训练...")
        obs, _ = self.env.reset()
        for key in obs:
            print(f"obs['{key}'].shape = {obs[key].shape}")
        # exit()
        
        for _ in range(total_steps):
            # 1. 代理采取动作
            action, _ = self.agent.predict(obs, deterministic=False)
            
            # 2. 与环境交互
            new_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 3. 存储经验并学习
            self.agent.learn(total_timesteps=1, reset_num_timesteps=False)
            
            # 4. 实时直播：每步都渲染或者每隔几步渲染一次（为了不拖慢训练速度）
            if self.global_step % 2 == 0:
                self.render_live(obs)
            
            obs = new_obs
            self.global_step += 1
            
            if terminated or truncated:
                obs, _ = self.env.reset()

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