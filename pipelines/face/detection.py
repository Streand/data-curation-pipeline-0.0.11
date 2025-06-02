import cv2
import numpy as np
import insightface
import torch
import time
import os
from utils.device import get_device


_face_model = None
_gpu_initialized = False

def get_face_model():
    """Get or initialize the face detection model"""
    global _face_model, _gpu_initialized
    
    if _face_model is None:
        # First try to initialize GPU properly
        if not _gpu_initialized and torch.cuda.is_available():
            _gpu_initialized()
        
        device = get_device()
        ctx_id = 0 if device == "cuda" else -1
        
        # Force CUDA provider to be first
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
        
        # Initialize with explicit provider order
        _face_model = insightface.app.FaceAnalysis(
            name="buffalo_l", 
            providers=providers
        )
        _face_model.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        print(f"Face detection model initialized on {device}")
        
        # Force model to load with a test image
        if torch.cuda.is_available():
            for _ in range(3):  # Try multiple times
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                mem_before = torch.cuda.memory_allocated() / (1024**2)
                
                dummy_img = np.ones((640, 640, 3), dtype=np.uint8) * 128
                _ = _face_model.get(dummy_img)
                
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated() / (1024**2)
                print(f"CUDA memory used: {mem_before:.1f} MB → {mem_after:.1f} MB (delta: {mem_after-mem_before:.1f} MB)")
                
                if mem_after - mem_before > 5.0:
                    break
                
                # If we're still not using GPU, try forcing it
                print("Trying again to force GPU usage...")
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    return _face_model

def detect_faces(image):
    """Detect faces in an image."""

    model = get_face_model()
    

    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            return []
    else:
        img = image
    

    try:

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / (1024**2)
            

            start_time = time.time()
            faces = model.get(img)
            torch.cuda.synchronize()
            
            end_time = time.time()
            mem_after = torch.cuda.memory_allocated() / (1024**2)
            print(f"Detection took {(end_time-start_time)*1000:.1f}ms, GPU memory delta: {mem_after-mem_before:.1f} MB")
        else:
            faces = model.get(img)
        

        return [{"bbox": face.bbox.tolist(), "score": face.det_score} for face in faces]
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []