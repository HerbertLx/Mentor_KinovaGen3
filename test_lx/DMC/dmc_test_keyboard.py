import os
import cv2
import numpy as np
import torch
from dm_control import suite
from dm_control import mujoco # 导入底层的 mujoco 模块用于视角控制

# 强制使用 glfw 进行渲染
os.environ['MUJOCO_GL'] = 'glfw' 

from stable_baselines3 import SAC
import gymnasium as gym
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

class Workspace:
    def __init__(self):
        self.domain_name = "dog"
        self.task_name = "walk"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 环境加载
        self.raw_env = suite.load(domain_name=self.domain_name, task_name=self.task_name)
        self.env = DmControlCompatibilityV0(self.raw_env, render_mode="rgb_array")
        self.env = gym.wrappers.RescaleAction(self.env, min_action=-1, max_action=1)
        
        # 2. 算法初始化
        self.agent = SAC("MultiInputPolicy", self.env, policy_kwargs=dict(net_arch=[512, 512, 512]), device=self.device)

        self.global_step = 0
        
        # --- 视角控制参数 ---
        self.cam_dist = 4.0      # 相机到目标的距离
        self.cam_azimuth = 90    # 水平旋转角度
        self.cam_elevation = -25  # 俯仰角 (负值代表从上往下看)
        self.lookat_offset = np.array([0.0, 0.0, 0.0]) # 目标点的微调偏移

    def render_live(self):
        """
        修正后的追踪渲染函数：使用标准的 Camera 接口
        """
        # 获取机械狗躯干的实时 3D 位置
        torso_pos = self.raw_env.physics.named.data.xpos['torso'].copy()
        
        # 核心修复：使用物理引擎自带的渲染功能并手动指定相机参数
        # 我们使用 camera_id=-1 配合自定义参数
        # 配置相机：盯着躯干中心 + 偏移量
        lookat = torso_pos + self.lookat_offset
        
        # 直接调用物理引擎渲染并控制视角
        frame = self.raw_env.physics.render(
            height=720, 
            width=1280, 
            camera_id=-1, # 使用自由相机
        )
        
        # 获取底层渲染上下文来修改相机姿态
        # 通过修改 physics.model 中的相机属性来实现
        # 这种方式比访问 render_contexts 更稳定
        self.raw_env.physics.model.stat.center[:] = lookat
        
        # 转换为 BGR 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 在屏幕上绘制 UI 信息
        overlay_color = (255, 255, 255)
        cv2.putText(frame, f"Step: {self.global_step}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, overlay_color, 2)
        cv2.putText(frame, "[WASD] Move LookAt | [ZX] Zoom | [Q] Quit", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2)
        
        cv2.imshow("Dog Walk - Pro Tracking", frame)
        
        # --- 键盘视角控制逻辑 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'): self.lookat_offset[0] += 0.2
        elif key == ord('a'): self.lookat_offset[0] -= 0.2
        elif key == ord('w'): self.lookat_offset[1] += 0.2
        elif key == ord('s'): self.lookat_offset[1] -= 0.2
        elif key == ord('z'): self.cam_dist = max(1.0, self.cam_dist - 0.5) # 放大
        elif key == ord('x'): self.cam_dist += 0.5 # 缩小
        elif key == ord('q'): exit()

    def train(self, total_steps=1000000):
        print("开始训练并开启实时追踪直播...")
        obs, _ = self.env.reset()
        
        for _ in range(total_steps):
            # 策略推理
            action, _ = self.agent.predict(obs, deterministic=False)
            
            # 环境步进
            new_obs, reward, terminated, truncated, _ = self.env.step(action)
            
            # 核心学习 (SAC 在 GPU 上运行)
            self.agent.learn(total_timesteps=1, reset_num_timesteps=False)
            
            # 实时渲染：为了保证追踪顺滑，每步都尝试更新画面
            if self.global_step % 2 == 0:
                self.render_live()
            
            obs = new_obs
            self.global_step += 1
            
            if terminated or truncated:
                obs, _ = self.env.reset()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    workspace = Workspace()
    workspace.train()