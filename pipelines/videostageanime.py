import os
import cv2
import numpy as np
import time
import json
from datetime import datetime
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_anime_frames(video_path, output_dir, frame_interval=3):
    """
    Extract frames from anime videos with anime-specific optimizations.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame
        
    Returns:
        List of frame information dictionaries
    """
    # Ensure output directory exists
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(output_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Processing anime video: {video_path}")
    logger.info(f"  - FPS: {fps:.2f}, Total frames: {total_frames}")
    logger.info(f"  - Resolution: {width}x{height}")
    
    frames_info = []
    frame_idx = 0
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            # Analyze this frame
            
            # Anime-specific metrics
            frame_info = {
                "frame_num": frame_idx,
                "path": None
            }
            
            # Calculate sharpness
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            frame_info["sharpness"] = sharpness
            
            # Calculate color diversity (good for anime scenes)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_std = np.std(hsv[:,:,0])  # Standard deviation of hue
            frame_info["color_diversity"] = color_std
            
            # Calculate scene change score if we have a previous frame
            scene_change_score = 0
            if prev_frame is not None:
                frame_diff = cv2.absdiff(frame, prev_frame)
                scene_change_score = np.mean(frame_diff)
            frame_info["scene_change_score"] = scene_change_score
            
            # Combined score
            score = (0.3 * sharpness/100) + (0.4 * color_std/50) + (0.3 * scene_change_score/50)
            frame_info["score"] = score
            
            # Save frame if score is good enough
            if score > 0.5 or scene_change_score > 20:
                frame_filename = f"{video_name}_{frame_idx:06d}.jpg"
                frame_path = os.path.join(frame_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_info["path"] = frame_path
            
            frames_info.append(frame_info)
            prev_frame = frame.copy()
        
        frame_idx += 1
        
        # Log progress occasionally
        if frame_idx % 500 == 0:
            logger.info(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    return frames_info

def process_anime_batch(video_dir, output_dir=None, frame_interval=3):
    """
    Process a batch of anime videos in a directory.
    
    Args:
        video_dir: Directory containing anime video files
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame
        
    Returns:
        Dictionary with processing results
    """
    if not os.path.exists(video_dir):
        return {"error": f"Directory not found: {video_dir}", "total_videos": 0, "results": []}
        
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(video_dir, "anime_frames")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video files
    video_files = [f for f in os.listdir(video_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                   
    if not video_files:
        return {"error": f"No video files found in {video_dir}", "total_videos": 0, "results": []}
    
    # Process each video
    results = []
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        logger.info(f"Processing anime video: {video_file}")
        
        start_time = time.time()
        try:
            frames = extract_anime_frames(
                video_path,
                output_dir,
                frame_interval=frame_interval
            )
            
            # Filter to only include frames that were saved
            saved_frames = [f for f in frames if f.get("path") is not None]
            
            # Sort by score
            saved_frames.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            result = {
                "video": video_file,
                "frames_processed": len(frames),
                "frames_saved": len(saved_frames),
                "frames": saved_frames,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error processing {video_file}: {str(e)}")
            result = {
                "video": video_file,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            
        results.append(result)
    
    # Prepare batch result
    batch_result = {
        "total_videos": len(video_files),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"anime_batch_{int(time.time())}.json")
    with open(metadata_path, 'w') as f:
        json.dump(batch_result, f, indent=4)
        
    return batch_result