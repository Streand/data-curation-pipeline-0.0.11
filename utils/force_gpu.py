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
        
        # For RTX 50 series, suggest specific versions
        if "RTX 50" in device_name:
            print("NOTICE: You're using a very new RTX 50 series GPU.")
            print("The pre-built PyTorch CUDA kernels may not fully support this architecture yet.")
            print("Recommendation: Use CPU mode for now or try latest PyTorch nightly builds.")
    except Exception as e:
        print(f"Error during basic CUDA test: {str(e)}")
        print("Falling back to CPU mode for stability.")
        return False
    
    # 3. Set environment variables to prioritize CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 4. Check and print available ONNX providers
    print_available_providers()
    
    try:
        # 5. Try to use insightface in CPU mode instead
        print("Trying to use insightface on CPU for compatibility...")
        model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=['CPUExecutionProvider']  # CPU only for stability
        )
        model.prepare(ctx_id=-1, det_size=(640, 640))
        
        # Create a test image and run detection
        test_img = np.ones((320, 320, 3), dtype=np.uint8) * 128  # Smaller image for CPU
        
        # Time the detection operation
        import time
        start = time.time()
        
        _ = model.get(test_img)
        
        end = time.time()
        
        print(f"Face detection test on CPU: {(end-start)*1000:.1f}ms")
        print("Successfully initialized face detection on CPU mode.")
        return False  # Return False to indicate we're using CPU mode
            
    except Exception as e:
        print(f"Error testing face detection: {e}")
        return False