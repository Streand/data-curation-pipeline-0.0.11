import cv2
import numpy as np
import insightface
import torch
import time
from utils.device import get_device


_face_model = None

def get_face_model():
    """Get or initialize the face detection model"""
    global _face_model
    if _face_model is None:
        device = get_device()
        ctx_id = 0 if device == "cuda" else -1
        

        _face_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
                                                  if device == "cuda" else ['CPUExecutionProvider'])
        

        _face_model.prepare(ctx_id=ctx_id, det_size=(640, 640))
        

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"Face detection model initialized on {device}")
            mem_before = torch.cuda.memory_allocated() / (1024**2)
            print(f"CUDA memory used: {mem_before:.1f} MB")
            

            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = _face_model.get(dummy_img)
            torch.cuda.synchronize()
            
            mem_after = torch.cuda.memory_allocated() / (1024**2)
            print(f"CUDA memory used after warmup: {mem_after:.1f} MB")
            print(f"Model memory footprint: {mem_after - mem_before:.1f} MB")
    
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