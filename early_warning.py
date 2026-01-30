import psutil
import os
import signal
import time
from pynput import keyboard

# --- 配置区 ---
MEMORY_THRESHOLD = 90.0  
CHECK_INTERVAL = 1.0     
# --------------
def on_press(key):
    if key == keyboard.Key.esc:
        kill_python_processes()

def get_python_procs():
    """获取当前所有 python 进程及其内存占用 (增强版)"""
    py_procs = []
    current_pid = os.getpid()
    # 增加 'cmdline' 属性
    for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cmdline']):
        try:
            # 1. 检查进程名
            name = (proc.info['name'] or "").lower()
            # 2. 检查完整的命令行（比如是否有 "python" 关键字）
            cmdline = " ".join(proc.info['cmdline'] or []).lower()
            
            if proc.info['pid'] != current_pid:
                if 'python' in name or 'python' in cmdline:
                    py_procs.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return py_procs

def kill_python_processes():
    """清理进程 (增强版)"""
    current_pid = os.getpid()
    print("\n" + "!"*40)
    print("[警告] 触发清理机制...")
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            name = (proc.info['name'] or "").lower()
            cmdline = " ".join(proc.info['cmdline'] or []).lower()
            
            if proc.info['pid'] != current_pid:
                if 'python' in name or 'python' in cmdline:
                    # 排除掉 VSCode 相关的基础插件进程（可选，如果你不想误伤 VSCode）
                    if 'extensions' in cmdline: 
                        continue
                        
                    print(f"终止 -> PID: {proc.info['pid']} | {cmdline[:50]}...")
                    os.kill(proc.info['pid'], signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    print("[完成] 清理完毕。")

# 启动键盘监听
listener = keyboard.Listener(on_press=on_press)
listener.start()

print(f"监控已启动 | 阈值: {MEMORY_THRESHOLD}% | ESC 紧急停止")

try:
    while True:
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        procs = get_python_procs()
        
        # 1. 清屏（保持界面整洁，Ubuntu 终端适用）
        print("\033[H\033[J", end="") 
        
        # 2. 显示系统总览
        print(f"--- 系统状态 ({time.strftime('%H:%M:%S')}) ---")
        print(f"总内存占用: {mem.percent}% {'[危险]' if mem.percent > MEMORY_THRESHOLD else ''}")
        print(f"总 CPU 占用: {cpu}%")
        print("-" * 40)
        
        # 3. 显示 Python 进程列表
        print(f"{'PID':<10} | {'内存 %':<10} | {'进程名称'}")
        if not procs:
            print("暂无运行中的 Python 实验进程")
        else:
            for p in procs:
                # 这里的 memory_percent 是指占总物理内存的比例
                print(f"{p['pid']:<10} | {p['memory_percent']:<10.2f} | {p['name']}")
        
        # 4. 阈值判断
        if mem.percent > MEMORY_THRESHOLD:
            kill_python_processes()
            time.sleep(2) # 停止后缓冲
            
        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print("\n监控已退出")
finally:
    listener.stop()