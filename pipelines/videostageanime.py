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

def extract_all_anime_frames(video_path, output_dir, frame_interval=1):
    """
    Stage 1: Extract ALL frames from anime videos without quality filtering
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (1 = every frame)
        
    Returns:
        List of frame information dictionaries
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(output_dir, video_name, "all_frames")
    os.makedirs(frame_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Extracting ALL frames from anime video: {video_path}")
    logger.info(f"  - FPS: {fps:.2f}, Total frames: {total_frames}")
    logger.info(f"  - Resolution: {width}x{height}")
    logger.info(f"  - Frame interval: {frame_interval} (extracting every {frame_interval} frame)")
    
    frames_info = []
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            # Save ALL frames without any quality filtering
            frame_filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
            frame_path = os.path.join(frame_dir, frame_filename)
            
            # Save the frame
            cv2.imwrite(frame_path, frame)
            
            frame_info = {
                "frame_num": frame_idx,
                "path": frame_path,
                "timestamp": frame_idx / fps if fps > 0 else 0,
                "saved": True
            }
            
            frames_info.append(frame_info)
            saved_count += 1
        
        frame_idx += 1
        
        # Progress logging
        if frame_idx % 1000 == 0:
            logger.info(f"Processed {frame_idx}/{total_frames} frames, saved {saved_count}")
    
    cap.release()
    
    # Save extraction metadata
    metadata = {
        "video_path": video_path,
        "video_name": video_name,
        "total_frames_in_video": total_frames,
        "frames_extracted": saved_count,
        "frame_interval": frame_interval,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "extraction_time": datetime.now().isoformat(),
        "output_directory": frame_dir
    }
    
    metadata_path = os.path.join(frame_dir, "extraction_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Extraction complete: {saved_count} frames saved to {frame_dir}")
    return frames_info

def filter_frames_by_anime_character(frames_dir, reference_char_path, similarity_threshold=0.6, max_results=1000):
    """
    Stage 2: Filter existing frames by anime character similarity
    
    Args:
        frames_dir: Directory containing extracted frames
        reference_char_path: Path to reference character image
        similarity_threshold: Minimum similarity threshold (0-1)
        max_results: Maximum number of matching frames to return
        
    Returns:
        Dictionary with filtering results
    """
    if not os.path.exists(reference_char_path):
        return {"error": "Reference character image not found"}
    
    if not os.path.exists(frames_dir):
        return {"error": "Frames directory not found"}
    
    # Load reference character
    ref_img = cv2.imread(reference_char_path)
    if ref_img is None:
        return {"error": "Could not load reference character image"}
    
    logger.info(f"Loading reference character from: {reference_char_path}")
    
    # Prepare reference features for anime character matching
    ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    
    # Use more bins for anime characters (they often have distinct colors)
    ref_hist = cv2.calcHist([ref_hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
    cv2.normalize(ref_hist, ref_hist, 0, 1, cv2.NORM_MINMAX)
    
    # Create output directory for character matches
    char_name = os.path.splitext(os.path.basename(reference_char_path))[0]
    char_dir = os.path.join(os.path.dirname(frames_dir), f"character_{char_name}")
    os.makedirs(char_dir, exist_ok=True)
    
    # Get all frame files
    frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    frame_files.sort()
    
    logger.info(f"Filtering {len(frame_files)} frames for anime character: {char_name}")
    logger.info(f"Similarity threshold: {similarity_threshold}")
    
    matches = []
    frames_processed = 0  # FIX: Initialize counter to track processed frames
    
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            continue
        
        frames_processed += 1  # FIX: Increment counter for each processed frame
        
        # Calculate similarity score for anime character
        similarity_score = calculate_anime_character_similarity(frame, ref_img, ref_hist)
        
        if similarity_score >= similarity_threshold:
            # Copy frame to character directory with similarity score in filename
            char_frame_path = os.path.join(char_dir, f"{char_name}_{similarity_score:.3f}_{frame_file}")
            cv2.imwrite(char_frame_path, frame)
            
            matches.append({
                "original_path": frame_path,
                "character_path": char_frame_path,
                "similarity_score": similarity_score,
                "frame_file": frame_file
            })
        
        # Progress logging
        if (i + 1) % 100 == 0:  # FIX: Use (i + 1) to avoid showing 0 processed frames
            logger.info(f"Processed {i + 1}/{len(frame_files)} frames, found {len(matches)} matches")
        
        # Early stop if we have enough matches
        if max_results and len(matches) >= max_results:
            logger.info(f"Reached maximum results limit ({max_results}), stopping early")
            break
    
    # Sort by similarity score (highest first)
    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Save filtering metadata
    metadata = {
        "reference_character": reference_char_path,
        "character_name": char_name,
        "frames_directory": frames_dir,
        "frames_processed": frames_processed,  # FIX: Use the dedicated counter
        "total_frames_available": len(frame_files),
        "matches_found": len(matches),
        "similarity_threshold": similarity_threshold,
        "filtering_time": datetime.now().isoformat(),
        "output_directory": char_dir
    }
    
    metadata_path = os.path.join(char_dir, "character_filtering_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Character filtering complete: {len(matches)} matches found")
    
    return {
        "matches": matches,
        "character_name": char_name,
        "total_matches": len(matches),
        "metadata": metadata,
        "character_directory": char_dir
    }

def calculate_anime_character_similarity(frame, ref_img, ref_hist):
    """
    Calculate similarity between frame and anime character reference
    Optimized for anime/cartoon characters with distinct colors and shapes
    """
    try:
        # Method 1: Enhanced histogram comparison for anime characters
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_hist = cv2.calcHist([frame_hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        cv2.normalize(frame_hist, frame_hist, 0, 1, cv2.NORM_MINMAX)
        
        hist_score = cv2.compareHist(ref_hist, frame_hist, cv2.HISTCMP_CORREL)
        
        # Method 2: Template matching (if reference is reasonable size)
        template_score = 0
        if ref_img.shape[0] < frame.shape[0]//2 and ref_img.shape[1] < frame.shape[1]//2:
            try:
                # Use normalized correlation for anime characters
                result = cv2.matchTemplate(frame, ref_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                template_score = max_val
            except:
                template_score = 0
        
        # Method 3: Color dominant region matching (good for anime characters)
        color_score = 0
        try:
            # Resize for faster processing
            frame_small = cv2.resize(frame, (200, 200))
            ref_small = cv2.resize(ref_img, (200, 200))
            
            # Calculate color distance
            frame_mean = np.mean(frame_small.reshape(-1, 3), axis=0)
            ref_mean = np.mean(ref_small.reshape(-1, 3), axis=0)
            
            color_distance = np.linalg.norm(frame_mean - ref_mean)
            color_score = max(0, 1 - (color_distance / 255.0))
        except:
            color_score = 0
        
        # Combine scores with weights optimized for anime characters
        # Histogram comparison is most reliable for anime characters
        final_score = (0.5 * hist_score) + (0.3 * template_score) + (0.2 * color_score)
        
        return max(0.0, min(1.0, final_score))
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

def process_anime_batch_stage1(video_dir, output_dir=None, frame_interval=1):
    """
    Stage 1: Process multiple anime videos to extract all frames
    """
    if not os.path.exists(video_dir):
        return {"error": f"Directory not found: {video_dir}", "total_videos": 0, "results": []}
        
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(video_dir, "anime_frames_stage1")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video files
    video_files = [f for f in os.listdir(video_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
                   
    if not video_files:
        return {"error": f"No video files found in {video_dir}", "total_videos": 0, "results": []}
    
    logger.info(f"Found {len(video_files)} anime videos to process")
    
    # Process each video
    results = []
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        logger.info(f"Processing anime video: {video_file}")
        
        start_time = time.time()
        try:
            frames = extract_all_anime_frames(
                video_path,
                output_dir,
                frame_interval=frame_interval
            )
            
            # Count saved frames
            saved_frames = [f for f in frames if f.get("saved", False)]
            
            result = {
                "video": video_file,
                "frames_extracted": len(saved_frames),
                "frame_interval": frame_interval,
                "frames": saved_frames,
                "processing_time": time.time() - start_time,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing {video_file}: {str(e)}")
            result = {
                "video": video_file,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "status": "error"
            }
            
        results.append(result)
    
    # Prepare batch result
    total_frames = sum(r.get("frames_extracted", 0) for r in results)
    batch_result = {
        "total_videos": len(video_files),
        "total_frames_extracted": total_frames,
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "frame_interval": frame_interval,
        "output_directory": output_dir,
        "stage": "1 - Frame Extraction"
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"anime_stage1_batch_{int(time.time())}.json")
    with open(metadata_path, 'w') as f:
        json.dump(batch_result, f, indent=4)
    
    logger.info(f"Stage 1 complete: {total_frames} frames extracted from {len(video_files)} videos")
    return batch_result