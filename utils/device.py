import torch
import os

_force_cpu = False

def set_force_cpu(value=True):
    """Force CPU usage regardless of GPU availability"""
    global _force_cpu
    _force_cpu = value

def get_device():
    """Get the appropriate device for model execution"""
    global _force_cpu
    
    if _force_cpu:
        return "cpu"
        
    if torch.cuda.is_available():
        try:
            # Test if CUDA is actually working
            x = torch.ones(1, device='cuda')
            y = x + x
            return "cuda"
        except Exception as e:
            print(f"CUDA error: {e}")
            print("Falling back to CPU")
            return "cpu"
    return "cpu"