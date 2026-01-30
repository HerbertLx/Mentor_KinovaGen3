import time
import os

def memory_test():
    print(f"--- 内存压力测试开始 (PID: {os.getpid()}) ---")
    print("提示：每秒将占用约 1GB 内存，观察监控脚本的反应。")
    
    data = []
    try:
        gb_count = 0
        while True:
            # 申请大约 1GB 的字节对象并添加到列表中，防止被垃圾回收
            # 10**9 字节约等于 1GB
            data.append(bytearray(10**9)) 
            gb_count += 1
            print(f"已占用: {gb_count} GB")
            time.sleep(1)
            
    except MemoryError:
        print("\n[错误] 内存已耗尽，脚本被系统/内核强制停止！")
    except KeyboardInterrupt:
        print("\n[停止] 用户手动停止了测试。")

if __name__ == "__main__":
    memory_test()