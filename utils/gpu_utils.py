# utils/gpu_utils.py

import pynvml
import torch

def find_free_gpu():
    """Tự động tìm và trả về ID của GPU có nhiều VRAM trống nhất."""
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            print("Không tìm thấy GPU nào.")
            return None

        best_gpu = -1
        max_free_mem = 0
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if mem_info.free > max_free_mem:
                max_free_mem = mem_info.free
                best_gpu = i
                
        pynvml.nvmlShutdown()

        if best_gpu != -1:
            print(f"Tìm thấy GPU rảnh nhất là: cuda:{best_gpu} với {max_free_mem / 1024**2:.2f} MB trống.")
            return best_gpu
        else:
            return 0 # Mặc định trả về GPU 0 nếu có lỗi
    except Exception as e:
        print(f"Lỗi khi tìm GPU, sử dụng GPU 0 mặc định. Lỗi: {e}")
        return 0