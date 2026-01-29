import gymnasium as gym
from stable_baselines3 import SAC
import time

# 1. 创建环境
# render_mode="human" 会打开一个窗口让你实时看到狗的行为
# dmc_proprio（本体感受）模式下，输入是关节角度等数值而非图片，训练极快
env = gym.make("shimmy/dog-walk-v0", render_mode="human")

print(f"动作空间: {env.action_space}") # 应该是 Box(-1.0, 1.0, (38,), float32)
print(f"观测空间: {env.observation_space}")

# 2. 定义 SAC 模型
# MlpPolicy 表示使用多层感知机（处理数值状态）
model = SAC(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    device="cuda" # 如果有GPU请设为cuda，否则cpu
)

# 3. 开始训练并实时渲染
print("开始训练... 你应该能看到弹出的 MuJoCo 窗口了。")
try:
    # 训练 100,000 步
    model.learn(total_timesteps=100000, log_interval=10)
    # 保存模型
    model.save("sac_dog_walk")
    print("训练完成并已保存模型。")
except KeyboardInterrupt:
    print("训练被用户中断。")

# 4. 训练结束后进行演示
print("开始演示训练结果...")
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render() # 持续渲染
    if terminated or truncated:
        obs, _ = env.reset()
    time.sleep(0.01)

env.close()