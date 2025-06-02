import cv2
import numpy as np
import insightface
import torch
from utils.device import get_device

# Initialize model only once (making it global)
_face_model = None

def get_face_model():
    """Get or initialize the face detection model"""
    global _face_model
    if _face_model is None:
        device = get_device()
        ctx_id = 0 if device == "cuda" else -1
        _face_model = insightface.app.FaceAnalysis()
        # Increase detection size for better accuracy
        _face_model.prepare(ctx_id=ctx_id, det_size=(640, 640))
        # Force CUDA synchronization if using GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"Face detection model initialized on {device}")
            print(f"CUDA memory used: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")
    return _face_model

def detect_faces(image):
    """Detect faces in an image.
    
    Args:
        image: Either a path to an image or a numpy array containing the image
    
    Returns:
        List of face detections, each with bbox and confidence score
    """
    # Get the model (initializes if needed)
    model = get_face_model()
    
    # Handle both image path and image array
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            return []
    else:
        img = image
    
    # Run face detection
    try:
        # Force CUDA synchronization before detection if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Detect faces
        faces = model.get(img)
        
        # Convert to simple dict format
        return [{"bbox": face.bbox.tolist(), "score": face.det_score} for face in faces]
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []