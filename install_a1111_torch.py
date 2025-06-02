import os
import sys
import subprocess
import re

# Clone A1111 repository to a temporary location if it doesn't exist
a1111_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_a1111")
if not os.path.exists(a1111_path):
    print("Cloning A1111 dev branch...")
    subprocess.check_call(["git", "clone", "-b", "dev", "https://github.com/AUTOMATIC1111/stable-diffusion-webui.git", a1111_path])

# Extract torch command from launch.py
launch_file = os.path.join(a1111_path, "launch.py")
torch_command = None

with open(launch_file, 'r', encoding='utf-8') as f:
    content = f.read()
    # Find torch_command variable
    match = re.search(r'torch_command\s*=\s*["\'](.+?)["\']', content)
    if match:
        torch_command = match.group(1)

if not torch_command:
    print("Could not find torch command in A1111 launch.py")
    sys.exit(1)

# Install PyTorch using A1111's command
print(f"Installing PyTorch using A1111's command: {torch_command}")

try:
    pip_command = [sys.executable, "-m", "pip", "install"] + torch_command.split()
    subprocess.check_call(pip_command)
    print("PyTorch installation complete!")
    
    # Install additional dependencies
    print("Installing additional dependencies for RTX 5080...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xformers==0.0.22"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "insightface", "onnxruntime-gpu"])
    
    # Test PyTorch installation
    print("\nTesting PyTorch installation...")
    test_cmd = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""
    subprocess.check_call([sys.executable, "-c", test_cmd])
    
except subprocess.CalledProcessError as e:
    print(f"Error during installation: {e}")
    sys.exit(1)