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

# Video selection presets
VIDEO_PRESETS = {
    "TikTok/Instagram": {
        "sample_rate": 1,
        "number_of_best_frames": 8,
        "use_clip_aesthetic": False,  # Changed to False
        "clip_prompt": "a high quality portrait photo, professional headshot",
        "filter_similar_frames": True,
        "scoring_weights": {
            "face_confidence": 0.35,
            "sharpness": 0.2,
            "aesthetic": 0.3,
            "face_size": 0.15
        },
        "thresholds": {
            "face_confidence": 0.90,
            "minimum_frame_distance": 15
        }
    },
    "YouTube": {
        "sample_rate": 5,
        "number_of_best_frames": 20,
        "use_clip_aesthetic": False,  # Changed to False
        "clip_prompt": "a high quality portrait photo, professional headshot",
        "filter_similar_frames": True,
        "scoring_weights": {
            "face_confidence": 0.3,
            "sharpness": 0.25,
            "aesthetic": 0.3,
            "face_size": 0.15
        },
        "thresholds": {
            "face_confidence": 0.88,
            "minimum_frame_distance": 30
        }
    },
    "Custom": {
        "sample_rate": 15,
        "number_of_best_frames": 20,
        "use_clip_aesthetic": False,  # Changed to False
        "clip_prompt": "a high quality portrait photo, professional headshot",
        "filter_similar_frames": True,
        "scoring_weights": {
            "face_confidence": 0.4,
            "sharpness": 0.2,
            "aesthetic": 0.2,
            "face_size": 0.2
        },
        "thresholds": {
            "face_confidence": 0.95,
            "minimum_frame_distance": 30
        }
    }
}

# Step 1: Add configurable thresholds
# Lower these values for more permissive selection
FACE_CONFIDENCE_THRESHOLD = 0.85  # Was 0.95
SHARPNESS_THRESHOLD = 50.0       # Was 100.0
CLIP_SCORE_THRESHOLD = 0.5        # Minimum CLIP similarity

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

# Step 2: Modify score_frames_for_quality with better composite scoring
# Update the function signature
def score_frames_for_quality(frame_paths, weights=None, thresholds=None, use_clip=None):
    device = get_device()
    results = []
    
    # Default weights if not provided
    if weights is None:
        weights = {
            "face_confidence": 0.4,
            "sharpness": 0.2,
            "face_size": 0.2,
            "aesthetic": 0.2
        }
    
    # Default thresholds if not provided
    if thresholds is None:
        thresholds = {
            "face_confidence": FACE_CONFIDENCE_THRESHOLD,
            "sharpness": SHARPNESS_THRESHOLD,
            "aesthetic": CLIP_SCORE_THRESHOLD
        }
    
    # Initialize CLIP model only if explicitly enabled
    clip_model = None
    clip_preprocess = None
    
    # Override use_clip value based on passed parameter
    if use_clip:
        try:
            clip_model, clip_preprocess = get_clip_model()
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"CLIP model loading failed: {e}")
            use_clip = False
            
            # Redistribute CLIP's weight to other factors if we can't use CLIP
            if "aesthetic" in weights and weights["aesthetic"] > 0:
                aesthetic_weight = weights["aesthetic"]
                weights["aesthetic"] = 0
                
                # Calculate total of remaining weights
                remaining_weights_total = sum(weights.values())
                
                # Distribute CLIP weight proportionally
                if remaining_weights_total > 0:  # Avoid division by zero
                    for key in weights:
                        if key != "aesthetic":
                            weights[key] += (weights[key] / remaining_weights_total) * aesthetic_weight
    else:
        print("CLIP scoring disabled")
        # Zero out aesthetic weight when CLIP is disabled
        weights["aesthetic"] = 0
        
        # Normalize remaining weights
        weight_sum = sum(weights.values())
        if weight_sum > 0:  # Avoid division by zero
            for key in weights:
                weights[key] = weights[key] / weight_sum
    
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
                    "has_face": False,
                    "passed_threshold": False
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
            
            # Calculate overall score with thresholds
            overall_score = 0
            passed_threshold = (max_face_score >= thresholds["face_confidence"] or sharpness >= thresholds["sharpness"])
            
            if passed_threshold:
                overall_score = (
                    weights["face_confidence"] * max_face_score +
                    weights["sharpness"] * (sharpness / 100) +
                    weights["face_size"] * (max_face_size / 10000) +
                    weights["aesthetic"] * aesthetic_score
                )
            
            results.append({
                "path": frame_path,
                "score": overall_score,
                "sharpness": sharpness,
                "face_score": max_face_score,
                "face_size": max_face_size,
                "aesthetic": aesthetic_score,
                "faces": len(faces),
                "has_face": True,
                "passed_threshold": passed_threshold
            })
            
        except Exception as e:
            results.append({
                "path": frame_path,
                "score": 0,
                "error": str(e),
                "passed_threshold": False
            })
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return results

# Step 3: Modify select_best_frames with improved de-duplication
# 1. Move get_dynamic_frame_count outside the function
def get_dynamic_frame_count(video_length_seconds, min_frames=5, max_frames=30):
    """Calculate number of frames to select based on video length"""
    return min(max(min_frames, int(video_length_seconds / 10)), max_frames)

# 2. Fix the function signature to include use_scene_detection
def select_best_frames(video_path, output_dir, preset="Custom", sample_rate=None, num_frames=None, 
                      use_clip=None, clip_prompt=None, min_frame_distance=None, use_scene_detection=True):
    """
    Process video and select best frames for training using preset or custom settings
    """
    # Load preset settings or use custom
    preset_config = VIDEO_PRESETS[preset]
    
    # Override preset with custom values if provided
    sample_rate = sample_rate if sample_rate is not None else preset_config["sample_rate"]
    num_frames = num_frames if num_frames is not None else preset_config["number_of_best_frames"]
    use_clip = use_clip if use_clip is not None else preset_config["use_clip_aesthetic"]
    clip_prompt = clip_prompt if clip_prompt is not None else preset_config["clip_prompt"]
    min_frame_distance = min_frame_distance if min_frame_distance is not None else preset_config["thresholds"]["minimum_frame_distance"]
    
    # Weights and thresholds from preset
    weights = preset_config["scoring_weights"]
    thresholds = preset_config["thresholds"]
    thresholds["sharpness"] = SHARPNESS_THRESHOLD  # Use global default or add to presets
    thresholds["aesthetic"] = CLIP_SCORE_THRESHOLD  # Use global default or add to presets
    
    # Check GPU usage
    print("Checking GPU status before processing...")
    check_gpu_usage()
    
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
    scored_frames = score_frames_for_quality(frames, weights=weights, thresholds=thresholds, use_clip=use_clip)
    
    # Boost scores for frames at scene changes
    if use_scene_detection and scene_frames:
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
    
    # Apply frame diversity selection
    diverse_frames = []
    selected_frame_numbers = []
    
    for frame_data in scored_frames:
        if not frame_data.get("passed_threshold", False):
            continue  # Skip frames that didn't pass quality thresholds
            
        frame_path = frame_data["path"]
        frame_num = int(os.path.basename(frame_path).split('_')[1].split('.')[0])
        
        # Check if this frame is far enough from already selected frames
        is_diverse = True
        
        for selected_num in selected_frame_numbers:
            if abs(frame_num - selected_num) < min_frame_distance:
                is_diverse = False
                break
        
        if is_diverse:
            diverse_frames.append(frame_data)
            selected_frame_numbers.append(frame_num)
            
            # Break if we have enough frames
            if len(diverse_frames) >= num_frames:
                break
    
    # If we don't have enough diverse frames, add best remaining frames
    if len(diverse_frames) < num_frames and len(scored_frames) > len(diverse_frames):
        remaining_needed = num_frames - len(diverse_frames)
        for frame in scored_frames:
            if frame not in diverse_frames and frame.get("score", 0) > 0:
                diverse_frames.append(frame)
                remaining_needed -= 1
                if remaining_needed <= 0:
                    break
    
    # Use fps and frame count to determine video length
    video_length = total_frames / fps
    if num_frames == 0:  # If auto mode
        num_frames = get_dynamic_frame_count(video_length)
    
    # Debug logging to help diagnose issues
    print(f"Total frames extracted: {total_frames}")
    print(f"Frames passing thresholds: {len([f for f in scored_frames if f.get('passed_threshold', False)])}")
    print(f"Diverse frames selected: {len(diverse_frames)}")
    print(f"Threshold values: Face confidence={thresholds['face_confidence']}, Sharpness={thresholds['sharpness']}")

    # FALLBACK: If no frames passed thresholds, relax criteria and try again
    if len(diverse_frames) == 0:
        print("WARNING: No frames passed thresholds, applying fallback selection...")
        
        # Sort by individual metrics instead
        best_faces = sorted(scored_frames, key=lambda x: x.get("face_score", 0), reverse=True)
        best_sharp = sorted(scored_frames, key=lambda x: x.get("sharpness", 0), reverse=True)
        
        # Take top frames from each criteria 
        fallback_frames = []
        for i in range(min(5, len(best_faces))):
            if best_faces[i] not in fallback_frames:
                fallback_frames.append(best_faces[i])
                
        for i in range(min(5, len(best_sharp))):
            if best_sharp[i] not in fallback_frames:
                fallback_frames.append(best_sharp[i])
                
        # Use as many as requested, up to what we have
        diverse_frames = fallback_frames[:num_frames]
        print(f"Fallback selected {len(diverse_frames)} frames")

    return diverse_frames, fps, total_frames

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

import os
import torch

def check_gpu_usage():
    """Check if GPU is being used and how much memory is allocated"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)    # MB
        
        print(f"Using: {device_name}")
        print(f"Memory allocated: {memory_allocated:.2f} MB")
        print(f"Memory reserved: {memory_reserved:.2f} MB")
        
        # Test if GPU is actually working
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            del x, y
            torch.cuda.empty_cache()
            print("GPU computation test: SUCCESS")
        except Exception as e:
            print(f"GPU computation test: FAILED - {e}")
    else:
        print("CUDA not available - using CPU")