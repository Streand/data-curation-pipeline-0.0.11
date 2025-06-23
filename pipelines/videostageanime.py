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

def extract_anime_frames(video_path, output_dir, frame_interval=3, reference_char=None, similarity_threshold=0.6):
    """
    Extract frames from anime videos with anime-specific optimizations.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame
        reference_char: Path to reference character image
        similarity_threshold: Minimum similarity threshold (0-1)
        
    Returns:
        List of frame information dictionaries
    """
    # Ensure output directory exists
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(output_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)
    
    # Load reference character if provided
    ref_features = None
    if reference_char and os.path.exists(reference_char):
        try:
            ref_img = cv2.imread(reference_char)
            if ref_img is not None:
                # Convert reference image to HSV for better color matching
                ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
                
                # Calculate histogram of reference image (for color distribution matching)
                ref_hist = cv2.calcHist([ref_hsv], [0, 1], None, [30, 30], [0, 180, 0, 256])
                cv2.normalize(ref_hist, ref_hist, 0, 1, cv2.NORM_MINMAX)
                
                # Store multiple feature types for better matching
                ref_features = {
                    'hist': ref_hist,
                    'image': ref_img,
                    'height': ref_img.shape[0],
                    'width': ref_img.shape[1]
                }
                logger.info(f"Loaded reference character image: {reference_char}")
        except Exception as e:
            logger.error(f"Error loading reference character: {str(e)}")
    
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
    
    # Create debug directory if reference character is used
    debug_dir = None
    if ref_features is not None:
        debug_dir = os.path.join(frame_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
    
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
            
            # Standard metrics (sharpness, color diversity, scene change)
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            frame_info["sharpness"] = sharpness
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_std = np.std(hsv[:,:,0])
            frame_info["color_diversity"] = color_std
            
            scene_change_score = 0
            if prev_frame is not None:
                frame_diff = cv2.absdiff(frame, prev_frame)
                scene_change_score = np.mean(frame_diff)
            frame_info["scene_change_score"] = scene_change_score
            
            # Character similarity calculation - improved version with multiple methods
            character_score = 0
            if ref_features is not None:
                # Method 1: Global histogram comparison
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame_hist = cv2.calcHist([frame_hsv], [0, 1], None, [30, 30], [0, 180, 0, 256])
                cv2.normalize(frame_hist, frame_hist, 0, 1, cv2.NORM_MINMAX)
                
                # Compare histograms using correlation method
                hist_score = cv2.compareHist(ref_features['hist'], frame_hist, cv2.HISTCMP_CORREL)
                
                # Method 2: Template matching
                # Resize reference image to reasonable search size if it's too large
                search_img = ref_features['image'].copy()
                max_search_dim = min(frame.shape[0]//2, frame.shape[1]//2)  # Don't use templates larger than half the frame
                
                if max(search_img.shape[0], search_img.shape[1]) > max_search_dim:
                    scale = max_search_dim / max(search_img.shape[0], search_img.shape[1])
                    search_img = cv2.resize(search_img, None, fx=scale, fy=scale)
                
                # Only do template matching if reference image is smaller than the frame
                template_score = 0
                if search_img.shape[0] < frame.shape[0] and search_img.shape[1] < frame.shape[1]:
                    # Use multiple matching methods and take the best score
                    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
                    best_score = 0
                    
                    for method in methods:
                        try:
                            result = cv2.matchTemplate(frame, search_img, method)
                            _, max_val, _, max_loc = cv2.minMaxLoc(result)
                            if max_val > best_score:
                                best_score = max_val
                                
                                # Save debug image if this is our best match so far
                                if debug_dir and best_score > template_score:
                                    debug_img = frame.copy()
                                    top_left = max_loc
                                    bottom_right = (top_left[0] + search_img.shape[1], top_left[1] + search_img.shape[0])
                                    cv2.rectangle(debug_img, top_left, bottom_right, (0, 255, 0), 2)
                                    debug_path = os.path.join(debug_dir, f"debug_{frame_idx:06d}_{best_score:.2f}.jpg")
                                    cv2.imwrite(debug_path, debug_img)
                        except Exception as e:
                            logger.error(f"Template matching error: {e}")
                    
                    template_score = best_score
                
                # Method 3: Region comparison (divide frame into regions and find best matching region)
                region_score = 0
                if frame.shape[0] > ref_features['height'] and frame.shape[1] > ref_features['width']:
                    ref_hsv = cv2.cvtColor(ref_features['image'], cv2.COLOR_BGR2HSV)
                    ref_hist_small = cv2.calcHist([ref_hsv], [0, 1], None, [20, 20], [0, 180, 0, 256])
                    cv2.normalize(ref_hist_small, ref_hist_small, 0, 1, cv2.NORM_MINMAX)
                    
                    # Divide frame into overlapping regions and compare
                    best_region_score = 0
                    step_size = max(frame.shape[0]//8, frame.shape[1]//8)  # Create overlapping regions
                    
                    # Limit the number of regions to check to avoid excessive computation
                    max_regions = 20
                    regions_checked = 0
                    
                    for y in range(0, frame.shape[0] - ref_features['height'], step_size):
                        if regions_checked >= max_regions:
                            break
                            
                        for x in range(0, frame.shape[1] - ref_features['width'], step_size):
                            if regions_checked >= max_regions:
                                break
                                
                            # Extract region
                            region = frame[y:y+ref_features['height'], x:x+ref_features['width']]
                            region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                            region_hist = cv2.calcHist([region_hsv], [0, 1], None, [20, 20], [0, 180, 0, 256])
                            cv2.normalize(region_hist, region_hist, 0, 1, cv2.NORM_MINMAX)
                            
                            # Compare histograms
                            region_corr = cv2.compareHist(ref_hist_small, region_hist, cv2.HISTCMP_CORREL)
                            
                            if region_corr > best_region_score:
                                best_region_score = region_corr
                                
                                # Save debug image for best region
                                if debug_dir and best_region_score > region_score + 0.1:  # Save only significant improvements
                                    debug_img = frame.copy()
                                    cv2.rectangle(debug_img, (x, y), (x+ref_features['width'], y+ref_features['height']), (0, 0, 255), 2)
                                    debug_path = os.path.join(debug_dir, f"region_{frame_idx:06d}_{best_region_score:.2f}.jpg")
                                    cv2.imwrite(debug_path, debug_img)
                            
                            regions_checked += 1
                    
                    region_score = best_region_score
                
                # Combine scores with weighted average - emphasize template and region matching which are more accurate
                character_score = (0.2 * hist_score) + (0.4 * template_score) + (0.4 * region_score)
                
                # Store individual scores for debugging
                frame_info["character_similarity"] = character_score
                frame_info["hist_score"] = hist_score
                frame_info["template_score"] = template_score  
                frame_info["region_score"] = region_score
                
                # Log detailed scores for troubleshooting
                logger.info(f"Frame {frame_idx}: character score={character_score:.3f}, hist={hist_score:.3f}, "
                          f"template={template_score:.3f}, region={region_score:.3f}")
            
            # Combined score calculation
            base_score = (0.3 * sharpness/100) + (0.4 * color_std/50) + (0.3 * scene_change_score/50)
            
            # Adjust score based on character presence
            if ref_features is not None:
                # Only consider frames with sufficient character similarity
                if character_score < similarity_threshold:
                    base_score *= 0.5  # Reduce score for frames without the character
                else:
                    # Boost score for frames with character
                    base_score *= (1.0 + character_score)
                    
            frame_info["score"] = base_score
            
            # Save frame if score is good enough AND character is present (if requested)
            save_frame = (base_score > 0.5 or scene_change_score > 20)
            
            # If using reference character, require minimum similarity
            # Use a dynamic threshold - if all scores are very low, gradually lower the threshold
            if ref_features is not None:
                # Lower the threshold a bit based on the frame index, to ensure we get at least some frames
                adaptive_threshold = max(similarity_threshold * 0.5, similarity_threshold - (frame_idx / total_frames * 0.3))
                save_frame = save_frame and (character_score >= adaptive_threshold)
                
            if save_frame:
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

def process_anime_batch(video_dir, output_dir=None, frame_interval=3, reference_char=None, similarity_threshold=0.6):
    """
    Process a batch of anime videos in a directory.
    
    Args:
        video_dir: Directory containing anime video files
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame
        reference_char: Path to reference character image
        similarity_threshold: Minimum similarity threshold (0-1)
        
    Returns:
        Dictionary with processing results
    """
    if not os.path.exists(video_dir):
        return {"error": f"Directory not found: {video_dir}", "total_videos": 0, "results": []}
        
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(video_dir, "anime_frames")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a folder for character-specific extraction if reference provided
    if reference_char and os.path.exists(reference_char):
        char_name = os.path.splitext(os.path.basename(reference_char))[0]
        output_dir = os.path.join(output_dir, f"character_{char_name}")
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
                frame_interval=frame_interval,
                reference_char=reference_char,
                similarity_threshold=similarity_threshold
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
        "timestamp": datetime.now().isoformat(),
        "reference_character": reference_char if reference_char else None,
        "similarity_threshold": similarity_threshold if reference_char else None
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"anime_batch_{int(time.time())}.json")
    with open(metadata_path, 'w') as f:
        json.dump(batch_result, f, indent=4)
        
    return batch_result