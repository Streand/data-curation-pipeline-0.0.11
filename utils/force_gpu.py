import os
import torch
import numpy as np
import insightface
import onnxruntime as ort

def print_available_providers():
    """Print all available ONNX Runtime providers"""
    providers = ort.get_available_providers()
    print(f"Available ONNX Runtime providers: {providers}")
    
    if 'CUDAExecutionProvider' not in providers:
        print("WARNING: CUDA provider not available in ONNX Runtime!")
        print("This is why face detection is falling back to CPU.")
        print("Try reinstalling onnxruntime-gpu with: pip install --force-reinstall onnxruntime-gpu")
    else:
        print("CUDA provider available for ONNX Runtime - good!")

def force_gpu_initialization():
    """Force proper GPU initialization for both PyTorch and ONNX Runtime"""
    if not torch.cuda.is_available():
        print("CUDA not available! Using CPU only.")
        return False
        
    # 1. Check CUDA device info
    try:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"PyTorch CUDA: {device_name} (Device {current_device}/{device_count-1})")
        
        # 2. Try a safer CUDA operation
        print("Testing basic CUDA operation...")
        x = torch.ones(10, device='cuda')
        y = x + x
        result = y.sum().item()
        print(f"Basic CUDA test: {result} (should be 20.0)")
        del x, y
        torch.cuda.empty_cache()
        
        # Check CUDA version compatibility
        cuda_version = torch.version.cuda
        print(f"PyTorch CUDA version: {cuda_version}")
        print(f"Your GPU: {device_name}")
        
    except Exception as e:
        print(f"Error during basic CUDA test: {str(e)}")
        print("Falling back to CPU mode for stability.")
        return False
    
    # 3. Set environment variables to prioritize CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 4. Check and print available ONNX providers
    print_available_providers()
    
    try:
        # 5. Try to use insightface with CUDA first
        print("Attempting to initialize insightface with CUDA...")
        model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        model.prepare(ctx_id=0, det_size=(640, 640))
        
        # Create a test image and run detection
        test_img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        
        # Check memory usage and time 
        mem_before = torch.cuda.memory_allocated() / (1024**2)
        
        import time
        torch.cuda.synchronize()
        start = time.time()
        
        _ = model.get(test_img)
        
        torch.cuda.synchronize()
        end = time.time()
        mem_after = torch.cuda.memory_allocated() / (1024**2)
        
        print(f"Face detection test on CUDA: {(end-start)*1000:.1f}ms")
        print(f"GPU memory usage: {mem_after-mem_before:.1f} MB")
        
        if mem_after - mem_before < 10.0:
            print("WARNING: GPU memory usage is suspiciously low. May still be using CPU.")
            
        return True
            
    except Exception as e:
        print(f"Error using GPU for face detection: {e}")
        print("Trying CPU fallback...")
        
        try:
            model = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=['CPUExecutionProvider']
            )
            model.prepare(ctx_id=-1, det_size=(640, 640))
            _ = model.get(test_img)
            print("Successfully initialized face detection on CPU mode.")
            return False
        except Exception as e2:
            print(f"CPU fallback also failed: {e2}")
            return False