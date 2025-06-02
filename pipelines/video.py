import cv2
import os
import numpy as np
from tqdm import tqdm
import torch
from utils.device import get_device
from pipelines.face.detection import detect_faces
from datetime import datetime
import json
import shutil

# Video selection presets
VIDEO_PRESETS = {
    "TikTok/Instagram": {
        "sample_rate": 1,
        "number_of_best_frames": 8,
        "filter_similar_frames": True,
        "scoring_weights": {
            "face_confidence": 0.6,
            "sharpness": 0.3,
            "face_size": 0.1
        },
        "thresholds": {
            "face_confidence": 0.90,
            "minimum_frame_distance": 15
        }
    },
    "YouTube": {
        "sample_rate": 5,
        "number_of_best_frames": 20,
        "filter_similar_frames": True,
        "scoring_weights": {
            "face_confidence": 0.5,
            "sharpness": 0.3,
            "face_size": 0.2
        },
        "thresholds": {
            "face_confidence": 0.88,
            "minimum_frame_distance": 30
        }
    },
    "Custom": {
        "sample_rate": 15,
        "number_of_best_frames": 20,
        "filter_similar_frames": True,
        "scoring_weights": {
            "face_confidence": 0.6,
            "sharpness": 0.3,
            "face_size": 0.1
        },
        "thresholds": {
            "face_confidence": 0.95,
            "minimum_frame_distance": 30
        }
    }
}

# Default thresholds - permissive for stage 1
FACE_CONFIDENCE_THRESHOLD = 0.7
SHARPNESS_THRESHOLD = 40.0

def get_app_root():
    """Get the application root directory in a portable way"""
    current_file = os.path.abspath(__file__)
    pipelines_dir = os.path.dirname(current_file)
    app_root = os.path.dirname(pipelines_dir)
    return app_root

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

def score_frames_for_stage1(frame_paths, weights=None, thresholds=None, progress=None, check_stop=None):
    """
    First-stage fast filtering: Uses only OpenCV and InsightFace
    with permissive thresholds to maximize recall.
    """
    device = get_device()
    results = []
    
    # Default weights if not provided - emphasize face detection and sharpness
    if weights is None:
        weights = {
            "face_confidence": 0.6,
            "sharpness": 0.3,
            "face_size": 0.1,
        }
    
    # Default thresholds if not provided - permissive
    if thresholds is None:
        thresholds = {
            "face_confidence": FACE_CONFIDENCE_THRESHOLD,
            "sharpness": SHARPNESS_THRESHOLD
        }
    
    # Use progress reporting if provided, otherwise fallback to tqdm
    total_frames = len(frame_paths)
    
    # Process each frame with progress reporting
    for i, frame_path in enumerate(frame_paths):
        try:
            # Report progress either to Gradio UI or tqdm
            if progress is not None:
                progress(i/total_frames, f"Fast analysis: {i}/{total_frames} frames")
            
            # Load image
            img = cv2.imread(frame_path)
            if img is None:
                raise Exception(f"Could not load image: {frame_path}")
            
            # Calculate sharpness (Laplacian variance)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Detect faces (InsightFace)
            faces = detect_faces(img)
            
            # If no faces found, still include but mark as no-face
            if not faces:
                results.append({
                    "path": frame_path,
                    "score": sharpness / 100,  # Simple score based on sharpness
                    "sharpness": sharpness,
                    "face_score": 0,
                    "face_size": 0,
                    "faces": 0,
                    "has_face": False,
                    "passed_threshold": sharpness >= thresholds["sharpness"]  # Still pass if sharp enough
                })
                continue
            
            # Find best face (highest confidence or largest)
            max_face_size = 0
            max_face_score = 0
            for face in faces:
                face_size = face["bbox"][2] * face["bbox"][3]
                if face_size > max_face_size:
                    max_face_size = face_size
                    max_face_score = face["score"]
            
            # PERMISSIVE THRESHOLD: Pass if EITHER good face OR good sharpness
            passed_threshold = (max_face_score >= thresholds["face_confidence"] or 
                               sharpness >= thresholds["sharpness"])
            
            # Calculate overall score (simpler, no CLIP)
            overall_score = (
                weights["face_confidence"] * max_face_score +
                weights["sharpness"] * (sharpness / 100) +
                weights["face_size"] * (max_face_size / 10000)
            )
            
            results.append({
                "path": frame_path,
                "score": overall_score,
                "sharpness": sharpness,
                "face_score": max_face_score,
                "face_size": max_face_size,
                "faces": len(faces),
                "has_face": True,
                "passed_threshold": passed_threshold,
                "stage": 1  # Mark as stage 1 processed
            })
            
        except Exception as e:
            results.append({
                "path": frame_path,
                "score": 0,
                "error": str(e),
                "passed_threshold": False,
                "stage": 1
            })
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return results

def select_frames_stage1(video_path, output_dir=None, preset="TikTok/Instagram", 
                       sample_rate=None, num_frames=None, min_frame_distance=30, 
                       use_scene_detection=False, progress=None, check_stop=None):
    """
    First-stage processing: Fast extraction of potential frames
    """
    # Default check_stop function if none provided
    if check_stop is None:
        check_stop = lambda: False
    
    # Get app root directory
    app_root = get_app_root()
    
    # Set up proper storage directory relative to the app root
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create storage directory under app root
    storage_dir = os.path.join(app_root, "store_images", "video_stage_1", f"{video_name}_{timestamp}")
    
    # Create the directory
    os.makedirs(storage_dir, exist_ok=True)
    
    # Use storage_dir as our output directory
    output_dir = storage_dir
    
    # Load preset settings
    preset_config = VIDEO_PRESETS[preset]
    
    # Override preset with custom values if provided
    sample_rate = sample_rate if sample_rate is not None else preset_config["sample_rate"]
    num_frames = num_frames if num_frames is not None else preset_config["number_of_best_frames"]
    min_frame_distance = min_frame_distance if min_frame_distance is not None else preset_config["thresholds"]["minimum_frame_distance"]
    
    # Weights and thresholds - simplified for stage 1
    weights = preset_config["scoring_weights"]
    
    thresholds = {
        "face_confidence": preset_config["thresholds"]["face_confidence"],  
        "sharpness": SHARPNESS_THRESHOLD
    }
    
    # Extract frames at given sample rate
    frames, fps = extract_frames(video_path, output_dir, sample_rate)
    
    if not frames:
        print("No frames were extracted from the video.")
        return [], fps, 0, None, None
    
    # Score frames with stage 1 function (fast)
    scored_frames = score_frames_for_stage1(frames, weights=weights, thresholds=thresholds, progress=progress, check_stop=check_stop)
    
    # Get passed frames
    passed_frames = [f for f in scored_frames if f.get("passed_threshold", False)]
    
    # FALLBACK: If no frames passed, take top N by score regardless
    if len(passed_frames) < min(5, num_frames):
        print(f"Only {len(passed_frames)} frames passed filters, adding more frames...")
        additional_needed = min(5, num_frames) - len(passed_frames)
        
        # Get frames that didn't pass but might be useful
        failed_frames = [f for f in scored_frames if not f.get("passed_threshold", False)]
        failed_frames.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Add the best of the failed frames
        for i in range(min(additional_needed, len(failed_frames))):
            failed_frames[i]["passed_threshold"] = True  # Mark as manually passed
            passed_frames.append(failed_frames[i])
    
    # Apply frame distance to ensure diversity
    diverse_frames = []
    selected_indices = []
    
    # Simplified scene detection - don't include the full SceneDetect dependency
    if use_scene_detection:
        print("Notice: Scene detection is disabled in Stage 1 for performance")
    
    # Select diverse frames based on distance
    for frame in passed_frames:
        frame_idx = frames.index(frame["path"])
        
        # Check if this frame is sufficiently distant from already selected frames
        if all(abs(frame_idx - selected) >= min_frame_distance for selected in selected_indices):
            diverse_frames.append(frame)
            selected_indices.append(frame_idx)
            
        # Stop if we have enough frames
        if len(diverse_frames) >= num_frames:
            break
    
    # If we don't have enough diverse frames, add best remaining frames
    if len(diverse_frames) < num_frames:
        remaining_frames = [f for f in passed_frames if f not in diverse_frames]
        remaining_frames.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        for frame in remaining_frames:
            if frame not in diverse_frames:
                diverse_frames.append(frame)
                
            if len(diverse_frames) >= num_frames:
                break
    
    # Debug info
    print(f"Stage 1 complete:")
    print(f"- Total frames extracted: {len(frames)}")
    print(f"- Frames passing thresholds: {len(passed_frames)}")
    print(f"- Diverse frames selected: {len(diverse_frames)}")
    
    # Save results metadata for later stages
    metadata_path = os.path.join(output_dir, "stage1_results.json")
    with open(metadata_path, 'w') as f:
        json.dump({
            "video_path": video_path,
            "fps": fps,
            "total_frames": len(frames),
            "passed_frames": len(passed_frames),
            "selected_frames": [
                {
                    "path": f["path"],
                    "score": f.get("score", 0),
                    "sharpness": f.get("sharpness", 0),
                    "face_score": f.get("face_score", 0),
                    "face_size": f.get("face_size", 0),
                    "faces": f.get("faces", 0)
                }
                for f in diverse_frames
            ]
        }, f, indent=2)
    
    # Copy best frames to a separate directory
    best_frames_dir = os.path.join(
        app_root, 
        "store_images", 
        "video_stage_1_best", 
        f"{video_name}_{timestamp}"
    )
    os.makedirs(best_frames_dir, exist_ok=True)

    # Copy each best frame to the new directory
    for frame in diverse_frames:
        try:
            src_path = frame["path"]
            filename = os.path.basename(src_path)
            # Add score to filename for easy sorting
            score_str = f"{frame.get('score', 0):.3f}".replace(".", "_")
            new_filename = f"score_{score_str}_{filename}"
            dst_path = os.path.join(best_frames_dir, new_filename)
            
            # Copy the file
            shutil.copy2(src_path, dst_path)
            
            # Update the path in the frame data to point to the new location
            frame["best_path"] = dst_path
        except Exception as e:
            print(f"Error copying best frame: {e}")

    # Add best frames directory to metadata
    with open(os.path.join(best_frames_dir, "best_frames_info.json"), 'w') as f:
        json.dump({
            "video_path": video_path,
            "fps": fps, 
            "total_frames": len(frames),
            "best_frames": diverse_frames
        }, f, indent=2)

    print(f"Best {len(diverse_frames)} frames copied to: {best_frames_dir}")

    # Return both directories in the results
    return diverse_frames, fps, len(frames), storage_dir, best_frames_dir