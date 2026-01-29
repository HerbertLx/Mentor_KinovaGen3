import os
import cv2
import torch
import numpy as np
import gymnasium as gym
from dm_control import suite
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# 1. 强制使用 glfw 以便弹出本地窗口
os.environ['MUJOCO_GL'] = 'glfw' 

class RealTimeMonitoringCallback(BaseCallback):
    """
    实时监控回调：
    1. 每一帧都进行渲染直播
    2. 实时在终端打印 Loss 和奖励
    """
    def __init__(self, raw_env, verbose=0):
        super(RealTimeMonitoringCallback, self).__init__(verbose)
        self.raw_env = raw_env
        self.fps_counter = 0
        # 相机控制参数
        self.cam_dist = 4.0
        self.lookat_offset = np.array([0.0, 0.0, 0.0])

    def _on_step(self) -> bool:
        # --- 实时渲染逻辑 ---
        # 获取机械狗躯干位置实现追踪
        torso_pos = self.raw_env.physics.named.data.xpos['torso'].copy()
        
        # 渲染大尺寸画面 (1280x720)
        # camera_id=1 通常是 DMC 默认的追踪相机，效果最稳
        frame = self.raw_env.physics.render(height=720, width=1280, camera_id=1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 提取训练指标
        logs = self.model.logger.name_to_value
        actor_loss = logs.get("train/actor_loss", 0)
        critic_loss = logs.get("train/loss", 0)
        reward = logs.get("rollout/ep_rew_mean", 0)

        # 在画面上绘制 HUD 信息
        cv2.putText(frame, f"Step: {self.num_timesteps}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Avg Reward: {reward:.2f}", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Actor Loss: {actor_loss:.4f}", (30, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("Dog Walk - Full-time Training Live", frame)

        # 处理键盘事件 (按 Q 退出)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False # 停止训练
        
        return True

def make_dmc_env():
    # 保持原生环境引用以便渲染
    raw_env = suite.load(domain_name="dog", task_name="walk")
    env = DmControlCompatibilityV0(raw_env, render_mode="rgb_array")
    env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
    return env, raw_env

if __name__ == "__main__":
    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"检测到设备: {device}")

    # 3. 创建环境
    env, raw_env = make_dmc_env()

    # 4. 定义模型 (针对 38 维动作空间优化网络规模)
    # 使用 MultiInputPolicy 处理字典格式的观测值
    model = SAC(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        buffer_size=300000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[512, 512, 512]),
        device=device
    )

    print("\n--- 训练说明 ---")
    print("1. 渲染窗口已开启，镜头将自动追踪机械狗。")
    print("2. 画面左上角实时显示步数、平均奖励和损失值。")
    print("3. 训练过程中按 'Q' 键可安全退出。")

    # 5. 开始全时段监控训练
    try:
        model.learn(
            total_timesteps=1000000,
            callback=RealTimeMonitoringCallback(raw_env)
        )
    except KeyboardInterrupt:
        print("手动停止训练")
    finally:
        cv2.destroyAllWindows()
        print("训练结束，窗口已关闭。")