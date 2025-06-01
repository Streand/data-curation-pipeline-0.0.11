import cv2
import os
import numpy as np
from tqdm import tqdm
import torch
from utils.device import get_device
from pipelines.face.detection import detect_faces

def extract_frames(video_path, output_dir, sample_rate=1):
    """
    Extract frames from video at given sample rate (1 = every frame, 2 = every other frame)
    Returns list of saved frame paths
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Calculate how many frames to skip
    frame_interval = int(sample_rate)
    
    saved_frames = []
    frame_idx = 0
    
    with tqdm(total=frame_count, desc="Extracting frames") as pbar:
        while True:
            success, frame = video.read()
            if not success:
                break
                
            if frame_idx % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_frames.append(frame_path)
            
            frame_idx += 1
            pbar.update(1)
    
    video.release()
    return saved_frames, fps

def score_frames_for_quality(frame_paths, use_cuda=True):
    """
    Score frames based on quality metrics:
    - Face detection confidence
    - Image sharpness
    - Face size
    """
    device = get_device()
    results = []
    
    for frame_path in tqdm(frame_paths, desc="Analyzing frames"):
        # Read image
        img = cv2.imread(frame_path)
        if img is None:
            continue
            
        # Measure sharpness using Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Detect faces using GPU if available
        try:
            _, faces = detect_faces(frame_path)  # detect_faces already uses CUDA if available
            
            # If no faces, give low score
            if not faces:
                results.append({
                    "path": frame_path,
                    "score": 0,
                    "sharpness": sharpness,
                    "faces": 0,
                    "has_face": False
                })
                continue
                
            # Calculate face size (largest face)
            max_face_size = 0
            max_face_score = 0
            for face in faces:
                bbox = face["bbox"]
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                face_size = face_width * face_height
                if face_size > max_face_size:
                    max_face_size = face_size
                    max_face_score = face["score"]
            
            # Calculate overall score (combination of metrics)
            # You can adjust weights here
            overall_score = (0.5 * max_face_score) + (0.3 * sharpness / 100) + (0.2 * max_face_size / 10000)
            
            results.append({
                "path": frame_path,
                "score": overall_score,
                "sharpness": sharpness,
                "face_score": max_face_score,
                "face_size": max_face_size,
                "faces": len(faces),
                "has_face": True
            })
            
        except Exception as e:
            results.append({
                "path": frame_path,
                "score": 0,
                "error": str(e)
            })
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results

def select_best_frames(video_path, output_dir, sample_rate=15, num_frames=20):
    """
    Process video and select best frames for training
    """
    # Extract frames at given sample rate
    frames, fps = extract_frames(video_path, output_dir, sample_rate)
    
    # Score frames
    scored_frames = score_frames_for_quality(frames)
    
    # Select top N frames
    best_frames = scored_frames[:num_frames] if len(scored_frames) > num_frames else scored_frames
    
    return best_frames, fps