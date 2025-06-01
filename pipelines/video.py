import cv2
import os
import numpy as np
from tqdm import tqdm
import torch
from utils.device import get_device
from pipelines.face.detection import detect_faces
import clip
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

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

def get_clip_model():
    device = get_device()
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def score_with_clip(img_path, model, preprocess, prompts=["a high quality portrait photo", "a professional headshot"]):
    device = get_device()
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    text = clip.tokenize(prompts).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    return similarity.mean().item()  # Average score across prompts

def score_frames_for_quality(frame_paths, use_cuda=True):
    device = get_device()
    results = []
    
    # Initialize CLIP model once (for efficiency)
    try:
        clip_model, clip_preprocess = get_clip_model()
        use_clip = True
    except Exception:
        print("CLIP not available, continuing without aesthetic scoring")
        use_clip = False
    
    # Track last frame's features for diversity calculation
    last_frame_features = None
    
    for frame_path in tqdm(frame_paths, desc="Analyzing frames"):
        # Read image
        img = cv2.imread(frame_path)
        if img is None:
            continue
            
        # Measure sharpness using Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Score components
        face_score = 0
        face_size = 0
        aesthetic_score = 0
        
        # Get clip score if available
        if use_clip:
            try:
                aesthetic_score = score_with_clip(frame_path, clip_model, clip_preprocess)
            except Exception as e:
                print(f"Error getting CLIP score: {e}")
                aesthetic_score = 0
        
        # Detect faces using GPU if available
        try:
            _, faces = detect_faces(frame_path)
            
            # If no faces, give low score
            if not faces:
                results.append({
                    "path": frame_path,
                    "score": 0,
                    "sharpness": sharpness,
                    "aesthetic": aesthetic_score,
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
            # Adjusted weights to include aesthetic score
            overall_score = (
                0.4 * max_face_score +          # Face detection confidence
                0.2 * (sharpness / 100) +       # Image sharpness
                0.2 * (max_face_size / 10000) + # Face size in frame
                0.2 * aesthetic_score           # CLIP aesthetic relevance
            )
            
            results.append({
                "path": frame_path,
                "score": overall_score,
                "sharpness": sharpness,
                "face_score": max_face_score,
                "face_size": max_face_size,
                "aesthetic": aesthetic_score,
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
    
    # Optional: Add diversity filtering (avoid similar frames)
    diverse_results = []
    for result in results:
        # Add logic here to check if frame is sufficiently different from already selected frames
        diverse_results.append(result)
    
    return results

def select_best_frames(video_path, output_dir, sample_rate=15, num_frames=20, use_clip=True, clip_prompt="a high quality portrait photo", use_scene_detection=True):
    """
    Process video and select best frames for training
    """
    # Extract frames at given sample rate
    frames, fps = extract_frames(video_path, output_dir, sample_rate)
    
    # Get total frames extracted
    total_frames = len(frames)
    
    # Get scene changes if enabled
    scene_frames = []
    if use_scene_detection:
        try:
            scene_frames = detect_scene_changes(video_path)
            print(f"Detected {len(scene_frames)} scene changes")
        except Exception as e:
            print(f"Error detecting scenes: {e}")
    
    # Score frames
    scored_frames = score_frames_for_quality(frames)
    
    # Boost scores for frames at scene changes
    if use_scene_detection and scene_frames:
        video = cv2.VideoCapture(video_path)
        for i, frame_data in enumerate(scored_frames):
            frame_path = frame_data["path"]
            # Extract frame number from filename (frame_000123.jpg)
            frame_num = int(os.path.basename(frame_path).split('_')[1].split('.')[0])
            
            # Boost score if this frame is near a scene change
            for scene_frame in scene_frames:
                if abs(frame_num - scene_frame) < 5:  # Within 5 frames of scene change
                    scored_frames[i]["score"] = scored_frames[i].get("score", 0) * 1.2  # 20% boost
                    break
        
        # Re-sort after boosting scene change frames
        scored_frames.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Select top N frames
    best_frames = scored_frames[:num_frames] if len(scored_frames) > num_frames else scored_frames
    
    return best_frames, fps, total_frames

def detect_scene_changes(video_path):
    """Detect major scene changes in video."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))  # Adjust threshold as needed
    
    # Detect scenes
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    # Extract frame numbers where scenes change
    scene_frames = [scene[0].get_frames() for scene in scene_list]
    
    return scene_frames