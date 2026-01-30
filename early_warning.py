import psutil
import os
import signal
import time
from pynput import keyboard

# --- 配置区 ---
MEMORY_THRESHOLD = 90.0  # 内存占用百分比阈值
CPU_THRESHOLD = 95.0     # CPU 占用百分比阈值（可选）
CHECK_INTERVAL = 1.0     # 监控频率（秒）
# --------------

def kill_python_processes():
    """找到并杀死除当前脚本外的所有 Python 进程"""
    current_pid = os.getpid()
    print("\n[警告] 触发清理机制，正在终止 Python 实验进程...")
    
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # 识别 Python 进程，且不是当前监控脚本
            if 'python' in proc.info['name'].lower() and proc.info['pid'] != current_pid:
                print(f"正在终止进程 PID: {proc.info['pid']} ({proc.info['name']})")
                os.kill(proc.info['pid'], signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    print("[完成] 所有实验进程已清理。\n")

def on_press(key):
    """监听键盘按键"""
    if key == keyboard.Key.esc:
        print("\n[手动触发] 检测到按下 ESC，准备终止进程...")
        kill_python_processes()
        # 如果你想让监控脚本也退出，可以取消下面这一行的注释
        # return False 

# 启动键盘监听（非阻塞模式）
listener = keyboard.Listener(on_press=on_press)
listener.start()

print(f"--- 实时监控已启动 ---")
print(f"设定阈值: 内存 > {MEMORY_THRESHOLD}%")
print(f"快捷键: 按下 ESC 立即终止所有 Python 进程")
print(f"{'时间':<10} | {'CPU %':<10} | {'内存 %':<10} | {'状态'}")
print("-" * 50)

try:
    while True:
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        mem_usage = memory_info.percent
        
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        status = "正常"
        
        # 实时显示
        print(f"{timestamp:<10} | {cpu_usage:<10.1f} | {mem_usage:<10.1f} | {status}", end='\r')
        
        # 逻辑判断
        if mem_usage > MEMORY_THRESHOLD:
            print(f"\n[触发] 内存占用 ({mem_usage}%) 超过阈值！")
            kill_python_processes()
            time.sleep(5) # 防止瞬间反复触发
            
        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print("\n监控脚本已停止。")
finally:
    listener.stop()