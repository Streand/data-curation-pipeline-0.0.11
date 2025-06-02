# Save as test_gpu.py
import torch
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Test CUDA operations
    print("\nRunning GPU performance test...")
    # Warm up
    x = torch.randn(3000, 3000, device=device)
    y = torch.randn(3000, 3000, device=device)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU matrix multiply time: {gpu_time:.4f} seconds")
    
    # Compare with CPU
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    start = time.time()
    z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"CPU matrix multiply time: {cpu_time:.4f} seconds")
    print(f"GPU is {cpu_time/gpu_time:.1f}x faster than CPU")
else:
    print("CUDA is not available. Check your installation.")