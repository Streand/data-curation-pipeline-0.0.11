import os
import cv2
import numpy as np
import time
import json
from datetime import datetime
import logging

# Use a placeholder NSFW detection function
# In a real implementation, you would use a proper NSFW detection model
def detect_nsfw_placeholder(image):
    """
    Placeholder function for NSFW detection.
    In real implementation, use a proper NSFW detection model.
    
    Args:
        image: Image to analyze
        
    Returns:
        NSFW score between 0 and 1
    """
    # This is just a placeholder that uses simple image statistics
    # A real implementation would use a proper model
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Simple placeholder using saturation and value levels
    saturation = np.mean(hsv[:,:,1]) / 255.0
    value = np.mean(hsv[:,:,2]) / 255.0
    
    # Generate somewhat random but deterministic score based on image content
    # Again, just a placeholder - real systems would use an actual NSFW model
    pixel_sum = np.sum(image) % 100 / 100
    
    # Combine factors for a placeholder score
    placeholder_score = (0.4 * saturation + 0.3 * value + 0.3 * pixel_sum)
    return placeholder_score

def extract_nsfw_frames(video_path, output_dir, nsfw_threshold=0.5, frame_sample_rate=10):
    """
    Extract frames from videos and analyze for NSFW content.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        nsfw_threshold: Threshold for NSFW classification
        frame_sample_rate: Analyze every Nth frame
        
    Returns:
        List of frame information dictionaries with NSFW scores
    """
    # Ensure output directory exists
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(output_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)
    
    # Create separate directories for safe and NSFW content
    safe_dir = os.path.join(frame_dir, "safe")
    nsfw_dir = os.path.join(frame_dir, "nsfw")
    os.makedirs(safe_dir, exist_ok=True)
    os.makedirs(nsfw_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_info = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_sample_rate == 0:
            # Analyze this frame for NSFW content
            nsfw_score = detect_nsfw_placeholder(frame)
            
            # Save frame info
            frame_info = {
                "frame_num": frame_idx,
                "nsfw_score": float(nsfw_score),
                "classification": "nsfw" if nsfw_score >= nsfw_threshold else "safe"
            }
            
            # Save the frame to appropriate directory
            if nsfw_score >= nsfw_threshold:
                save_dir = nsfw_dir
            else:
                save_dir = safe_dir
                
            frame_filename = f"{video_name}_{frame_idx:06d}_{nsfw_score:.3f}.jpg"
            frame_path = os.path.join(save_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_info["path"] = frame_path
            
            frames_info.append(frame_info)
        
        frame_idx += 1
        
        # Log progress occasionally
        if frame_idx % 500 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    return frames_info

def process_nsfw_analysis(video_dir, output_dir=None, nsfw_threshold=0.5, frame_sample_rate=10):
    """
    Process videos to detect and classify NSFW content.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save analyzed frames
        nsfw_threshold: Threshold for NSFW classification
        frame_sample_rate: Analyze every Nth frame
        
    Returns:
        Dictionary with analysis results
    """
    if not os.path.exists(video_dir):
        return {"error": f"Directory not found: {video_dir}", "total_videos": 0, "results": []}
        
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(video_dir, "nsfw_analysis")
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
        print(f"Processing video for NSFW content: {video_file}")
        
        start_time = time.time()
        try:
            frames = extract_nsfw_frames(
                video_path,
                output_dir,
                nsfw_threshold=nsfw_threshold,
                frame_sample_rate=frame_sample_rate
            )
            
            # Count frames by classification
            safe_frames = [f for f in frames if f.get("classification") == "safe"]
            nsfw_frames = [f for f in frames if f.get("classification") == "nsfw"]
            
            result = {
                "video": video_file,
                "frames_analyzed": len(frames),
                "safe_frames": len(safe_frames),
                "nsfw_frames": len(nsfw_frames),
                "nsfw_percentage": len(nsfw_frames) / len(frames) if frames else 0,
                "frames": frames,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
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
        "timestamp": datetime.now().isoformat(),
        "nsfw_threshold": nsfw_threshold
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"nsfw_analysis_{int(time.time())}.json")
    with open(metadata_path, 'w') as f:
        json.dump(batch_result, f, indent=4)
        
    return batch_result