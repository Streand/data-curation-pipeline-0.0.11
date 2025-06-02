import insightface
import cv2
import torch
import numpy as np
import time
import os

# Disable memory optimization
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Create a directory for test images
os.makedirs("test_images", exist_ok=True)

# Generate standard-sized images that are compatible with RetinaFace
def create_test_image(index, size=(1920, 1080)):  # Larger resolution
    img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    # Add some face-like patterns to trigger detection
    for i in range(20):  # More patterns
        x = np.random.randint(0, size[0]-300)
        y = np.random.randint(0, size[1]-300)
        cv2.ellipse(img, (x+150, y+150), (100, 130), 0, 0, 360, (220, 180, 160), -1)
        cv2.circle(img, (x+100, y+100), 25, (40, 40, 100), -1)
        cv2.circle(img, (x+200, y+100), 25, (40, 40, 100), -1)
        cv2.ellipse(img, (x+150, y+180), (50, 25), 0, 0, 360, (180, 100, 100), -1)
    
    path = f"test_images/test_image_{index}.jpg"
    cv2.imwrite(path, img)
    return path

print("Generating test images...")
image_paths = [create_test_image(i) for i in range(15)]  # More images

# Create a large persistent tensor that stays in memory throughout execution
print("\n===== PHASE 0: BASELINE MEMORY ALLOCATION =====")
persistent_tensor = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
print(f"Baseline tensor allocated, VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Heavy PyTorch operations to stress GPU memory
print("\n===== PHASE 1: HEAVY PYTORCH OPERATIONS =====")
dummy_tensors = []

# Create larger tensors
for i in range(5):  # More tensors
    size = 7000 + (i * 1000)  # Larger sizes
    print(f"Creating large tensor {i+1}/5 ({size}×{size})")
    tensor = torch.randn(size, size, device='cuda', dtype=torch.float32)
    dummy_tensors.append(tensor)
    print(f"  Current VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Matrix multiplications (very GPU intensive)
print("\nRunning matrix multiplications...")
for i in range(8):  # More operations
    print(f"Matrix operation {i+1}/8")
    # Pick different tensor combinations for each operation
    a = dummy_tensors[i % len(dummy_tensors)]
    b = dummy_tensors[(i+1) % len(dummy_tensors)]
    # Calculate only on a slice to avoid CUDA OOM errors
    slice_size = min(4500, a.size(0), b.size(0))  # Larger slice
    result = torch.matmul(a[:slice_size, :slice_size], b[:slice_size, :slice_size])
    dummy_tensors.append(result)
    print(f"  Current VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Load more models and larger ones
print("\n===== PHASE 2: LOADING MULTIPLE MODELS =====")
models = []

# Load a mix of different models to maximize VRAM usage
model_types = ['resnet152', 'densenet201', 'efficientnet_b7', 'resnet101']
for i, model_type in enumerate(model_types):
    print(f"Loading {model_type} model ({i+1}/{len(model_types)})")
    try:
        model = torch.hub.load('pytorch/vision:v0.10.0', model_type, pretrained=True)
        model = model.cuda()
        model.eval()
        # Make weights different to prevent sharing
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        models.append(model)
        print(f"  Current VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    except Exception as e:
        print(f"  Failed to load {model_type}: {e}")

# Use insightface with safe image sizes
print("\n===== PHASE 3: FACE DETECTION ON MULTIPLE IMAGES =====")
app = insightface.app.FaceAnalysis(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0)  # Using the default size (640×640) which is safe

# Hold results in memory to increase usage
all_faces = []
all_embeddings = []
feature_tensors = []

# Process images multiple times with larger batch sizes
for j in range(4):  # More loops
    print(f"\nProcessing loop {j+1}/4:")
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        print(f"  Image {i+1}/{len(image_paths)} - shape: {img.shape}")
        faces = app.get(img)
        all_faces.extend(faces)  # Keep all faces in memory
        print(f"  Found {len(faces)} faces (total: {len(all_faces)})")
        print(f"  Current VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # Extract feature maps from multiple models with larger batches
        if models and i % 2 == 0:  # Process more frequently
            # Create larger batch size for inference
            batch_size = 48  # Increased from 16
            input_tensor = torch.randn(batch_size, 3, 224, 224).cuda()
            for model_idx, model in enumerate(models):
                print(f"  Running inference on model {model_idx+1}/{len(models)}")
                with torch.no_grad():
                    features = model(input_tensor)
                    feature_tensors.append(features)  # Store outputs
            print(f"  After inference VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Optional: create a few more tensors to fill remaining VRAM
print("\n===== PHASE 4: FINAL MEMORY PUSH =====")
leftover_tensors = []
for i in range(10):  # Create several small ones
    size = 2000
    tensor = torch.randn(size, size, device='cuda', dtype=torch.float32)
    leftover_tensors.append(tensor)
    print(f"Additional tensor {i+1}/10, VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

print("\n===== FINAL MEMORY STATS =====")
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
print(f"Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f}GB")
print("Holding memory for 20 seconds for monitoring...")
time.sleep(20)
print("Test complete!")