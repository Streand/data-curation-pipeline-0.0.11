import os
import sys
import subprocess
import re
import shutil
from pathlib import Path

def main():
    print("RTX 5080 PyTorch Installation Helper")
    print("-----------------------------------")
    
    # Create temp directory for A1111 clone
    temp_dir = Path("./temp_a1111")
    if temp_dir.exists():
        print(f"Cleaning previous installation files...")
        shutil.rmtree(temp_dir)
    
    temp_dir.mkdir(exist_ok=True)
    
    # Clone A1111 dev branch
    print("Cloning A1111 dev branch (this may take a minute)...")
    try:
        subprocess.check_call(["git", "clone", "-b", "dev", "--depth=1", 
                              "https://github.com/AUTOMATIC1111/stable-diffusion-webui.git", 
                              str(temp_dir)])
    except Exception as e:
        print(f"Error cloning A1111: {e}")
        return False
    
    # Extract torch command from modules/launch_utils.py
    launch_utils_path = temp_dir / "modules" / "launch_utils.py"
    
    if not launch_utils_path.exists():
        print(f"Error: Cannot find {launch_utils_path}")
        return False
    
    # Get torch command from A1111's code
    torch_command = None
    with open(launch_utils_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Look for torch commands specific to newer GPUs
        for line in content.split('\n'):
            if 'torch_command =' in line and ('nightly' in line or 'dev' in line):
                match = re.search(r'torch_command\s*=\s*["\'](.+?)["\']', line)
                if match:
                    torch_command = match.group(1)
                    print(f"Found GPU-optimized torch command: {torch_command}")
                    break
        
        # If no specialized command found, get the default one
        if not torch_command:
            match = re.search(r'torch_command\s*=\s*["\'](.+?)["\']', content)
            if match:
                torch_command = match.group(1)
                print(f"Using default torch command: {torch_command}")
    
    if not torch_command:
        print("Error: Could not find torch installation command")
        return False
    
    # Install PyTorch
    print("\nInstalling PyTorch for RTX 5080...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pip"])
        
        # Execute the torch command
        if torch_command.startswith("pip "):
            torch_command = torch_command[4:]
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + torch_command.split())
        
        print("PyTorch installation complete!")
    except Exception as e:
        print(f"Error during PyTorch installation: {e}")
        return False
    
    # Check for any specific environment settings in A1111 code
    print("\nChecking for needed environment variables...")
    try:
        devices_path = temp_dir / "modules" / "devices.py"
        if devices_path.exists():
            with open(devices_path, 'r', encoding='utf-8') as f:
                devices_content = f.read()
                # Look for environment variable settings for RTX cards
                env_vars = re.findall(r'os\.environ\[[\'"](.*?)[\'"]\]\s*=\s*[\'"](.+?)[\'"]', devices_content)
                for var, value in env_vars:
                    if any(x in var for x in ['CUDA', 'TORCH']):
                        print(f"Setting {var}={value}")
                        os.environ[var] = value
    except Exception as e:
        print(f"Warning: Could not process environment variables: {e}")
    
    # Install xformers for efficiency
    print("\nInstalling xformers...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xformers==0.0.22", "--prefer-binary"])
    except Exception as e:
        print(f"Warning: xformers installation failed: {e}")
    
    # Reinstall insightface for face detection
    print("\nInstalling insightface with GPU support...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "insightface", "onnxruntime-gpu"])
    except Exception as e:
        print(f"Warning: insightface installation failed: {e}")
    
    # Run GPU test
    print("\nTesting GPU configuration...")
    try:
        subprocess.check_call([sys.executable, "test_gpu.py"])
    except Exception as e:
        print(f"Warning: GPU test failed: {e}")
        return False
    
    print("\n✅ RTX 5080 setup complete!")
    print("You can now run your video processing with GPU acceleration.")
    
    # Cleanup
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    return True

if __name__ == "__main__":
    main()