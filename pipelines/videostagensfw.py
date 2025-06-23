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

def init_nudenet():
    try:
        from nudenet import NudeDetector
        detector = NudeDetector()
        logger.info("NudeNet model loaded successfully")
        return detector
    except Exception as e:
        logger.error(f"Error loading NudeNet model: {str(e)}")
        return None

# Global model instances
NUDE_DETECTOR = init_nudenet()

def detect_comprehensive_nsfw(image):
    """
    Comprehensive NSFW detection focusing on your specific requirements:
    - Nudity (exposed breasts, buttocks, genitals)
    - Revealing clothing (cleavage, tight clothes, see-through)
    - Suggestive poses and actions
    - Covered nipples but visible through clothing
    """
    global NUDE_DETECTOR
    
    if NUDE_DETECTOR is None:
        return {"score": 0.0, "detections": [], "categories": []}
    
    try:
        # Save image temporarily for NudeNet processing
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"temp_nsfw_{int(time.time())}_{np.random.randint(10000)}.jpg")
        
        cv2.imwrite(temp_path, image)
        
        # Run NudeNet detection
        detections = NUDE_DETECTOR.detect(temp_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        # FIXED: More balanced NSFW categories
        nsfw_categories = {
            # High priority - explicit nudity ONLY
            'EXPOSED_ANUS': 1.0,
            'EXPOSED_FEMALE_GENITALIA': 1.0,
            'EXPOSED_MALE_GENITALIA': 1.0,
            'EXPOSED_BREAST_F': 0.9,  # Exposed female breasts
            'EXPOSED_BUTTOCKS': 0.8,   # Exposed buttocks
            
            # Medium-high priority - suggestive but covered
            'FEMALE_BREAST_COVERED': 0.4,  # REDUCED from 0.7 to 0.4 - denim vest shouldn't be high priority
            'FEMALE_GENITALIA_COVERED': 0.5,  # Underwear, tight clothing
            'BUTTOCKS_COVERED': 0.4,   # Tight clothing, underwear
            
            # Medium priority - revealing clothing
            'EXPOSED_BELLY': 0.3,      # Exposed midriff, crop tops
            'EXPOSED_ARMPITS': 0.2,    # Tank tops, sleeveless
            
            # Low priority - less revealing
            'EXPOSED_BREAST_M': 0.1,   # Male chest
            'MALE_BREAST_COVERED': 0.05,
            'MALE_GENITALIA_COVERED': 0.2,
            'BELLY_COVERED': 0.1,
            'ARMPITS_COVERED': 0.05,
            
            # Neutral categories
            'FACE_F': 0.0,
            'FACE_M': 0.0,
            'FEET_EXPOSED': 0.05,
            'FEET_COVERED': 0.0,
        }
        
        total_score = 0.0
        detected_categories = []
        high_priority_found = False
        
        for detection in detections:
            category = detection['class']
            confidence = detection['score']
            
            if category in nsfw_categories:
                category_weight = nsfw_categories[category]
                contribution = confidence * category_weight
                total_score += contribution
                
                # FIXED: Only truly explicit content counts as high priority
                if category_weight >= 0.8:  # Changed from 0.7 to 0.8
                    high_priority_found = True
                
                detected_categories.append({
                    'category': category,
                    'confidence': confidence,
                    'weight': category_weight,
                    'contribution': contribution,
                    'bbox': detection['box'],
                    'priority': 'high' if category_weight >= 0.8 else 'medium' if category_weight >= 0.3 else 'low'
                })
        
        # Additional scoring for revealing clothing patterns
        revealing_clothing_score = detect_revealing_patterns(image)
        total_score += revealing_clothing_score * 0.2  # REDUCED from 0.3 to 0.2
        
        # FIXED: More conservative score boosting - only for truly explicit content
        if high_priority_found:
            # Check if it's truly explicit (exposed genitals/breasts)
            explicit_categories = ['EXPOSED_ANUS', 'EXPOSED_FEMALE_GENITALIA', 'EXPOSED_MALE_GENITALIA', 'EXPOSED_BREAST_F']
            truly_explicit = any(cat['category'] in explicit_categories for cat in detected_categories)
            if truly_explicit:
                total_score *= 1.3  # REDUCED from 1.5 to 1.3
        
        # Normalize score to 0-1 range
        final_score = min(total_score, 1.0)
        
        return {
            "score": final_score,
            "detections": detections,
            "categories": detected_categories,
            "raw_score": total_score,
            "high_priority_found": high_priority_found,
            "revealing_clothing_score": revealing_clothing_score
        }
        
    except Exception as e:
        logger.error(f"Error in NSFW detection: {str(e)}")
        return {"score": 0.0, "detections": [], "categories": []}

def detect_revealing_patterns(image):
    """
    Detect revealing clothing patterns like:
    - High skin exposure
    - Tight clothing outlines
    - See-through materials
    - Cleavage areas
    """
    try:
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Skin tone detection - multiple ranges for different skin tones
        skin_masks = []
        
        # Light skin tones
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        skin_masks.append(cv2.inRange(hsv, lower_skin1, upper_skin1))
        
        # Medium skin tones
        lower_skin2 = np.array([0, 25, 50], dtype=np.uint8)
        upper_skin2 = np.array([15, 255, 200], dtype=np.uint8)
        skin_masks.append(cv2.inRange(hsv, lower_skin2, upper_skin2))
        
        # Combine all skin masks
        combined_skin_mask = skin_masks[0]
        for mask in skin_masks[1:]:
            combined_skin_mask = cv2.bitwise_or(combined_skin_mask, mask)
        
        # Calculate skin exposure metrics
        total_pixels = image.shape[0] * image.shape[1]
        skin_pixels = cv2.countNonZero(combined_skin_mask)
        skin_percentage = skin_pixels / total_pixels
        
        # Analyze skin distribution
        contours, _ = cv2.findContours(combined_skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        revealing_score = 0.0
        
        # High skin exposure indicator
        if skin_percentage > 0.25:  # More than 25% skin visible
            revealing_score += min(skin_percentage * 2, 0.8)
        
        # Multiple skin regions (indicates gaps in clothing)
        if len(contours) > 3:
            revealing_score += min(len(contours) * 0.1, 0.4)
        
        # Large continuous skin areas (bikinis, crop tops, etc.)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > total_pixels * 0.08:  # Large skin area
                revealing_score += min(area / total_pixels, 0.3)
        
        return min(revealing_score, 1.0)
        
    except Exception as e:
        logger.error(f"Error in revealing pattern detection: {str(e)}")
        return 0.0

def extract_nsfw_frames(video_path, output_dir, nsfw_threshold=0.4, frame_interval=10, draw_boxes=True):
    """
    Extract frames with NSFW content from videos.
    Lower threshold = more strict (detects more as NSFW)
    
    Args:
        draw_boxes: If True, draw bounding boxes around detected NSFW areas
    """
    # Ensure output directory exists
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(output_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)
    
    # Create separate directories for different NSFW levels
    high_nsfw_dir = os.path.join(frame_dir, "high_nsfw")
    medium_nsfw_dir = os.path.join(frame_dir, "medium_nsfw")
    low_nsfw_dir = os.path.join(frame_dir, "low_nsfw")
    safe_dir = os.path.join(frame_dir, "safe")
    
    for dir_path in [high_nsfw_dir, medium_nsfw_dir, low_nsfw_dir, safe_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Processing NSFW video: {video_path}")
    logger.info(f"  - FPS: {fps:.2f}, Total frames: {total_frames}")
    
    frames_info = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            # Analyze frame for NSFW content
            nsfw_result = detect_comprehensive_nsfw(frame)
            nsfw_score = nsfw_result["score"]
            
            # FIXED: More appropriate classification thresholds
            if nsfw_score >= 0.9:  # INCREASED from 0.8 to 0.9 - only truly explicit
                classification = "high_nsfw"
                save_dir = high_nsfw_dir
            elif nsfw_score >= 0.6:  # Medium remains the same
                classification = "medium_nsfw" 
                save_dir = medium_nsfw_dir
            elif nsfw_score >= nsfw_threshold:  # Low NSFW
                classification = "low_nsfw"
                save_dir = low_nsfw_dir
            else:
                classification = "safe"
                save_dir = safe_dir
            
            # Create a copy of the frame for drawing boxes
            frame_with_boxes = frame.copy()
            
            # Draw bounding boxes if requested and NSFW content detected
            if draw_boxes and nsfw_score >= nsfw_threshold:
                frame_with_boxes = draw_nsfw_boxes(frame_with_boxes, nsfw_result)
            
            # Save frame info
            frame_info = {
                "frame_num": frame_idx,
                "nsfw_score": float(nsfw_score),
                "classification": classification,
                "categories": nsfw_result["categories"],
                "high_priority_found": nsfw_result["high_priority_found"],
                "boxes_drawn": draw_boxes and nsfw_score >= nsfw_threshold
            }
            
            # Save the frame (with or without boxes)
            frame_filename = f"{video_name}_{frame_idx:06d}_{nsfw_score:.3f}.jpg"
            frame_path = os.path.join(save_dir, frame_filename)
            cv2.imwrite(frame_path, frame_with_boxes)
            frame_info["path"] = frame_path
            
            frames_info.append(frame_info)
        
        frame_idx += 1
        
        # Log progress occasionally
        if frame_idx % 1000 == 0:
            logger.info(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    return frames_info

def draw_nsfw_boxes(frame, nsfw_result):
    """
    Draw bounding boxes around detected NSFW areas with color coding.
    
    Args:
        frame: The image frame to draw on
        nsfw_result: Result from detect_comprehensive_nsfw
        
    Returns:
        Frame with bounding boxes drawn
    """
    # Color coding for different priority levels
    colors = {
        'high': (0, 0, 255),      # Red for high priority (explicit)
        'medium': (0, 165, 255),  # Orange for medium priority  
        'low': (0, 255, 255),     # Yellow for low priority
        'safe': (0, 255, 0)       # Green for safe (shouldn't happen but just in case)
    }
    
    # Thickness based on priority
    thickness = {
        'high': 3,
        'medium': 2,
        'low': 1,
        'safe': 1
    }
    
    frame_with_boxes = frame.copy()
    
    # Draw boxes for each detected category
    for category_info in nsfw_result.get("categories", []):
        bbox = category_info.get("bbox", [])
        priority = category_info.get("priority", "low")
        category = category_info.get("category", "unknown")
        confidence = category_info.get("confidence", 0)
        
        if len(bbox) == 4:  # [x, y, width, height]
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Draw rectangle
            color = colors.get(priority, (255, 255, 255))  # Default to white
            thick = thickness.get(priority, 1)
            
            cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), color, thick)
            
            # Add label with category and confidence
            label = f"{category}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background rectangle for text
            cv2.rectangle(frame_with_boxes, 
                         (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), 
                         color, -1)
            
            # Text
            cv2.putText(frame_with_boxes, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add overall score in top-left corner
    overall_score = nsfw_result.get("score", 0)
    score_text = f"NSFW Score: {overall_score:.3f}"
    cv2.putText(frame_with_boxes, score_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame_with_boxes

def process_nsfw_batch(video_dir, output_dir=None, nsfw_threshold=0.4, frame_interval=10, draw_boxes=True):
    """
    Process a batch of videos for NSFW content detection.
    
    Args:
        draw_boxes: If True, draw bounding boxes around detected NSFW areas
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
        logger.info(f"Processing video for NSFW content: {video_file}")
        
        start_time = time.time()
        try:
            frames = extract_nsfw_frames(
                video_path,
                output_dir,
                nsfw_threshold=nsfw_threshold,
                frame_interval=frame_interval,
                draw_boxes=draw_boxes  # Pass the draw_boxes parameter
            )
            
            # Count frames by classification
            high_nsfw = [f for f in frames if f.get("classification") == "high_nsfw"]
            medium_nsfw = [f for f in frames if f.get("classification") == "medium_nsfw"]
            low_nsfw = [f for f in frames if f.get("classification") == "low_nsfw"]
            safe_frames = [f for f in frames if f.get("classification") == "safe"]
            
            result = {
                "video": video_file,
                "frames_analyzed": len(frames),
                "high_nsfw_frames": len(high_nsfw),
                "medium_nsfw_frames": len(medium_nsfw),
                "low_nsfw_frames": len(low_nsfw),
                "safe_frames": len(safe_frames),
                "nsfw_percentage": (len(high_nsfw) + len(medium_nsfw) + len(low_nsfw)) / len(frames) if frames else 0,
                "frames": frames,
                "processing_time": time.time() - start_time,
                "boxes_drawn": draw_boxes
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
        "nsfw_threshold": nsfw_threshold,
        "boxes_drawn": draw_boxes
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"nsfw_batch_{int(time.time())}.json")
    with open(metadata_path, 'w') as f:
        json.dump(batch_result, f, indent=4)
        
    return batch_result