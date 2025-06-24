import os
import cv2
import numpy as np
import time
import json
from datetime import datetime
import logging
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn.functional as F
from typing import Optional, Union

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import functions from stage 1
from pipelines.videostageanime import calculate_anime_character_similarity

# FIXED: Anime character-specific feature extractor with proper typing
class AnimeCharacterExtractor:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing anime character extractor on: {self.device}")
        
        # Use ResNet50 but focus on character-specific features
        self.model = resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        # Anime-optimized preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # FIXED: Initialize reference data with proper types
        self.reference_image: Optional[np.ndarray] = None
        self.ref_hist: Optional[np.ndarray] = None
    
    def extract_anime_features(self, image_batch):
        """Extract features optimized for anime character recognition"""
        with torch.no_grad():
            features = self.model(image_batch)
            features = features.flatten(1)
            features = F.normalize(features, p=2, dim=1)
        return features
    
    def process_image(self, image):
        """Process image with anime character focus"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB for anime processing
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image)
    
    def set_reference_data(self, ref_img: np.ndarray) -> None:
        """FIXED: Properly set reference image and histogram"""
        self.reference_image = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
        self.ref_hist = cv2.calcHist([ref_hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        # FIXED: Proper cv2.normalize call
        self.ref_hist = cv2.normalize(self.ref_hist, self.ref_hist, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

def calculate_anime_character_similarity_gpu(reference_features, frame_batch, extractor):
    """
    FIXED: Calculate anime character similarity using multiple methods
    """
    try:
        # Method 1: Deep feature similarity
        frame_tensors = []
        for frame in frame_batch:
            frame_tensor = extractor.process_image(frame)
            frame_tensors.append(frame_tensor)
        
        batch_tensor = torch.stack(frame_tensors).to(extractor.device)
        frame_features = extractor.extract_anime_features(batch_tensor)
        
        # Deep feature similarity
        deep_similarities = torch.mm(reference_features, frame_features.t())
        deep_similarities = deep_similarities.squeeze().cpu().numpy()
        
        # Method 2: Traditional anime character matching for validation
        traditional_similarities = []
        ref_cpu = cv2.cvtColor(extractor.reference_image, cv2.COLOR_RGB2BGR) if extractor.reference_image is not None else None
        
        for frame in frame_batch:
            if ref_cpu is not None and extractor.ref_hist is not None:
                # Use the original anime character similarity function
                similarity = calculate_anime_character_similarity(frame, ref_cpu, extractor.ref_hist)
                traditional_similarities.append(similarity)
            else:
                traditional_similarities.append(0.5)
        
        # Combine both methods (weighted average)
        if isinstance(deep_similarities, np.ndarray):
            if deep_similarities.ndim == 0:
                deep_similarities = [deep_similarities.item()]
            else:
                deep_similarities = deep_similarities.tolist()
        
        # Weighted combination: 60% deep features, 40% traditional anime matching
        final_similarities = []
        for i in range(len(frame_batch)):
            deep_sim = deep_similarities[i] if i < len(deep_similarities) else 0.5
            trad_sim = traditional_similarities[i] if i < len(traditional_similarities) else 0.5
            
            # Combine with weights optimized for anime characters
            final_sim = (0.6 * deep_sim) + (0.4 * trad_sim)
            final_similarities.append(max(0.0, min(1.0, final_sim)))
        
        return final_similarities
        
    except Exception as e:
        logger.error(f"GPU anime character similarity error: {e}")
        return [0.3] * len(frame_batch)  # Lower default for stricter matching

def process_character_filtering_batch_gpu(frames_base_dir, reference_char_path, similarity_threshold=0.6, max_results_per_video=None, batch_size=32):
    """
    FIXED: GPU-accelerated anime character filtering with proper character detection
    """
    if not os.path.exists(frames_base_dir):
        return {"error": f"Frames directory not found: {frames_base_dir}"}
    
    if not os.path.exists(reference_char_path):
        return {"error": f"Reference character image not found: {reference_char_path}"}
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU processing")
        return process_character_filtering_batch(frames_base_dir, reference_char_path, similarity_threshold, max_results_per_video)
    
    # Log GPU info
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using GPU for anime character detection: {gpu_name} ({gpu_memory:.1f}GB)")
    except Exception as e:
        logger.warning(f"Could not get GPU info: {e}")
    
    # FIXED: Initialize anime character extractor
    extractor = AnimeCharacterExtractor()
    
    # FIXED: Process reference character with anime-specific methods
    ref_img = cv2.imread(reference_char_path)
    if ref_img is None:
        return {"error": "Could not load reference character image"}
    
    # FIXED: Use the new method to set reference data
    extractor.set_reference_data(ref_img)
    
    # Extract deep features for reference
    try:
        ref_tensor = extractor.process_image(ref_img).unsqueeze(0).to(extractor.device)
        ref_features = extractor.extract_anime_features(ref_tensor)
    except Exception as e:
        logger.error(f"Error processing reference image: {e}")
        return {"error": f"Error processing reference image: {e}"}
    
    # Find video frame directories
    video_frame_dirs = []
    for item in os.listdir(frames_base_dir):
        item_path = os.path.join(frames_base_dir, item)
        if os.path.isdir(item_path):
            all_frames_dir = os.path.join(item_path, "all_frames")
            if os.path.exists(all_frames_dir):
                video_frame_dirs.append({
                    "video_name": item,
                    "frames_dir": all_frames_dir,
                    "video_dir": item_path
                })
    
    if not video_frame_dirs:
        return {"error": "No video frame directories found. Please run Stage 1 first."}
    
    char_name = os.path.splitext(os.path.basename(reference_char_path))[0]
    logger.info(f"Starting anime character detection for '{char_name}' across {len(video_frame_dirs)} videos")
    logger.info(f"Similarity threshold: {similarity_threshold} (stricter = fewer matches)")
    
    # Create output directory
    output_dir = os.path.join(frames_base_dir, f"character_{char_name}_matches")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total_matches = 0
    
    for video_info in video_frame_dirs:
        video_name = video_info["video_name"]
        frames_dir = video_info["frames_dir"]
        
        logger.info(f"Processing anime character detection for video: {video_name}")
        
        start_time = time.time()
        try:
            frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            frame_files.sort()
            
            if not frame_files:
                logger.warning(f"No frame files found in {frames_dir}")
                continue
            
            matches = []
            frames_processed = 0
            
            # Process in batches
            for i in range(0, len(frame_files), batch_size):
                batch_files = frame_files[i:i+batch_size]
                batch_frames = []
                batch_paths = []
                
                # Load batch
                for frame_file in batch_files:
                    frame_path = os.path.join(frames_dir, frame_file)
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        batch_frames.append(frame)
                        batch_paths.append(frame_path)
                
                if not batch_frames:
                    continue
                
                # FIXED: Use anime character similarity
                similarities = calculate_anime_character_similarity_gpu(ref_features, batch_frames, extractor)
                
                # Process results with stricter filtering
                for similarity_score, frame_path, frame in zip(similarities, batch_paths, batch_frames):
                    frames_processed += 1
                    
                    # FIXED: Only save frames that actually match the character
                    if similarity_score >= similarity_threshold:
                        char_video_dir = os.path.join(output_dir, f"character_{char_name}")
                        os.makedirs(char_video_dir, exist_ok=True)
                        
                        frame_file = os.path.basename(frame_path)
                        char_frame_path = os.path.join(char_video_dir, f"{char_name}_{similarity_score:.3f}_{frame_file}")
                        cv2.imwrite(char_frame_path, frame)
                        
                        matches.append({
                            "original_path": frame_path,
                            "character_path": char_frame_path,
                            "similarity_score": float(similarity_score),
                            "frame_file": frame_file
                        })
                
                # Progress logging
                if (i + batch_size) % (batch_size * 20) == 0 or (i + batch_size) >= len(frame_files):
                    match_rate = (len(matches) / frames_processed * 100) if frames_processed > 0 else 0
                    logger.info(f"Processed {min(i + batch_size, len(frame_files))}/{len(frame_files)} frames, found {len(matches)} character matches ({match_rate:.1f}% match rate)")
                
                # FIXED: Handle unlimited results properly
                if max_results_per_video is not None and len(matches) >= max_results_per_video:
                    logger.info(f"Reached maximum results limit ({max_results_per_video}), stopping early")
                    break
                
                # FIXED: Clear GPU memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Sort by similarity score
            matches.sort(key=lambda x: x["similarity_score"], reverse=True)
            total_matches += len(matches)
            
            # Calculate match rate for logging
            match_rate = (len(matches) / frames_processed * 100) if frames_processed > 0 else 0
            
            result = {
                "video": video_name,
                "matches_found": len(matches),
                "frames_processed": frames_processed,
                "match_rate_percent": round(match_rate, 2),
                "character_directory": os.path.join(output_dir, f"character_{char_name}"),
                "matches": matches[:20],  # Preview only
                "processing_time": time.time() - start_time,
                "gpu_accelerated": True,
                "status": "success"
            }
            
            logger.info(f"Video '{video_name}': {len(matches)} character matches found ({match_rate:.1f}% of {frames_processed} frames)")
            
        except Exception as e:
            logger.error(f"Error processing {video_name}: {str(e)}")
            result = {
                "video": video_name,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "status": "error"
            }
        
        results.append(result)
        
        # FIXED: Clear GPU cache after each video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Overall statistics
    total_frames_processed = sum(r.get("frames_processed", 0) for r in results if r.get("status") == "success")
    overall_match_rate = (total_matches / total_frames_processed * 100) if total_frames_processed > 0 else 0
    
    batch_result = {
        "character_name": char_name,
        "reference_character_path": reference_char_path,
        "total_videos_processed": len(video_frame_dirs),
        "total_frames_processed": total_frames_processed,
        "total_character_matches": total_matches,
        "overall_match_rate_percent": round(overall_match_rate, 2),
        "similarity_threshold": similarity_threshold,
        "gpu_accelerated": True,
        "gpu_info": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        },
        "batch_size": batch_size,
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "output_directory": output_dir,
        "stage": "2 - GPU Anime Character Detection"
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"anime_character_detection_{int(time.time())}.json")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(batch_result, f, indent=4)
    except Exception as e:
        logger.warning(f"Could not save metadata: {e}")
    
    logger.info(f"GPU anime character detection complete:")
    logger.info(f"  - {total_matches} character matches found across {len(video_frame_dirs)} videos")
    logger.info(f"  - Overall match rate: {overall_match_rate:.1f}% ({total_matches}/{total_frames_processed} frames)")
    logger.info(f"  - Results saved to: {output_dir}")
    
    return batch_result

# FIXED: Proper CPU fallback function
def process_character_filtering_batch(frames_base_dir, reference_char_path, similarity_threshold=0.6, max_results_per_video=None):
    """
    FIXED: CPU fallback for character filtering when GPU is not available
    """
    if not os.path.exists(frames_base_dir):
        return {"error": f"Frames directory not found: {frames_base_dir}"}
    
    if not os.path.exists(reference_char_path):
        return {"error": f"Reference character image not found: {reference_char_path}"}
    
    logger.info("Using CPU for anime character detection")
    
    # Find video frame directories
    video_frame_dirs = []
    for item in os.listdir(frames_base_dir):
        item_path = os.path.join(frames_base_dir, item)
        if os.path.isdir(item_path):
            all_frames_dir = os.path.join(item_path, "all_frames")
            if os.path.exists(all_frames_dir):
                video_frame_dirs.append({
                    "video_name": item,
                    "frames_dir": all_frames_dir,
                    "video_dir": item_path
                })
    
    if not video_frame_dirs:
        return {"error": "No video frame directories found. Please run Stage 1 first."}
    
    char_name = os.path.splitext(os.path.basename(reference_char_path))[0]
    logger.info(f"Starting CPU anime character detection for '{char_name}' across {len(video_frame_dirs)} videos")
    
    # Load reference image
    ref_img = cv2.imread(reference_char_path)
    if ref_img is None:
        return {"error": "Could not load reference character image"}
    
    # FIXED: Create reference histogram with proper cv2.normalize
    ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    ref_hist = cv2.calcHist([ref_hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
    ref_hist = cv2.normalize(ref_hist, ref_hist, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    
    # Create output directory
    output_dir = os.path.join(frames_base_dir, f"character_{char_name}_matches")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total_matches = 0
    
    for video_info in video_frame_dirs:
        video_name = video_info["video_name"]
        frames_dir = video_info["frames_dir"]
        
        logger.info(f"Processing CPU character detection for video: {video_name}")
        
        start_time = time.time()
        try:
            frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            frame_files.sort()
            
            if not frame_files:
                logger.warning(f"No frame files found in {frames_dir}")
                continue
            
            matches = []
            frames_processed = 0
            
            for frame_file in frame_files:
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                
                if frame is not None:
                    frames_processed += 1
                    
                    # Use original anime character similarity function
                    similarity_score = calculate_anime_character_similarity(frame, ref_img, ref_hist)
                    
                    if similarity_score >= similarity_threshold:
                        char_video_dir = os.path.join(output_dir, f"character_{char_name}")
                        os.makedirs(char_video_dir, exist_ok=True)
                        
                        char_frame_path = os.path.join(char_video_dir, f"{char_name}_{similarity_score:.3f}_{frame_file}")
                        cv2.imwrite(char_frame_path, frame)
                        
                        matches.append({
                            "original_path": frame_path,
                            "character_path": char_frame_path,
                            "similarity_score": float(similarity_score),
                            "frame_file": frame_file
                        })
                
                # Progress logging
                if frames_processed % 1000 == 0:
                    match_rate = (len(matches) / frames_processed * 100) if frames_processed > 0 else 0
                    logger.info(f"Processed {frames_processed}/{len(frame_files)} frames, found {len(matches)} character matches ({match_rate:.1f}% match rate)")
                
                # Handle unlimited results
                if max_results_per_video is not None and len(matches) >= max_results_per_video:
                    logger.info(f"Reached maximum results limit ({max_results_per_video}), stopping early")
                    break
            
            # Sort by similarity score
            matches.sort(key=lambda x: x["similarity_score"], reverse=True)
            total_matches += len(matches)
            
            match_rate = (len(matches) / frames_processed * 100) if frames_processed > 0 else 0
            
            result = {
                "video": video_name,
                "matches_found": len(matches), 
                "frames_processed": frames_processed,
                "match_rate_percent": round(match_rate, 2),
                "character_directory": os.path.join(output_dir, f"character_{char_name}"),
                "matches": matches[:20],  # Preview only
                "processing_time": time.time() - start_time,
                "gpu_accelerated": False,
                "status": "success"
            }
            
            logger.info(f"Video '{video_name}': {len(matches)} character matches found ({match_rate:.1f}% of {frames_processed} frames)")
            
        except Exception as e:
            logger.error(f"Error processing {video_name}: {str(e)}")
            result = {
                "video": video_name,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "status": "error"
            }
        
        results.append(result)
    
    # Overall statistics
    total_frames_processed = sum(r.get("frames_processed", 0) for r in results if r.get("status") == "success")
    overall_match_rate = (total_matches / total_frames_processed * 100) if total_frames_processed > 0 else 0
    
    batch_result = {
        "character_name": char_name,
        "reference_character_path": reference_char_path,
        "total_videos_processed": len(video_frame_dirs),
        "total_frames_processed": total_frames_processed,
        "total_character_matches": total_matches,
        "overall_match_rate_percent": round(overall_match_rate, 2),
        "similarity_threshold": similarity_threshold,
        "gpu_accelerated": False,
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "output_directory": output_dir,
        "stage": "2 - CPU Anime Character Detection"
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"anime_character_detection_{int(time.time())}.json")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(batch_result, f, indent=4)
    except Exception as e:
        logger.warning(f"Could not save metadata: {e}")
    
    logger.info(f"CPU anime character detection complete:")
    logger.info(f"  - {total_matches} character matches found across {len(video_frame_dirs)} videos")
    logger.info(f"  - Overall match rate: {overall_match_rate:.1f}% ({total_matches}/{total_frames_processed} frames)")
    
    return batch_result

def get_available_extracted_frames(base_dir):
    """Get available frame directories"""
    available_videos = []
    
    if not os.path.exists(base_dir):
        return available_videos
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            all_frames_dir = os.path.join(item_path, "all_frames")
            if os.path.exists(all_frames_dir):
                frame_count = len([f for f in os.listdir(all_frames_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                available_videos.append({
                    "video_name": item,
                    "frames_directory": all_frames_dir,
                    "frame_count": frame_count,
                    "full_path": item_path
                })
    
    return available_videos

def preview_character_matches(character_matches_dir, max_preview=20):
    """Get preview of character matches"""
    if not os.path.exists(character_matches_dir):
        return []
    
    char_dirs = []
    
    # Check for images in main directory
    if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(character_matches_dir)):
        char_dirs.append(character_matches_dir)
    
    # Check subdirectories
    for item in os.listdir(character_matches_dir):
        item_path = os.path.join(character_matches_dir, item)
        if os.path.isdir(item_path):
            char_dirs.append(item_path)
    
    preview_items = []
    
    for char_dir in char_dirs:
        image_files = [f for f in os.listdir(char_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sort by similarity score
        def extract_score(filename):
            try:
                parts = filename.split('_')
                for part in parts:
                    if part.startswith('0.') and len(part) <= 6:  # FIXED: Allow for 0.xxx format
                        return float(part)
                return 0.0
            except:
                return 0.0
        
        image_files.sort(key=extract_score, reverse=True)
        
        for img_file in image_files[:max_preview]:
            img_path = os.path.join(char_dir, img_file)
            score = extract_score(img_file)
            char_dir_name = os.path.basename(char_dir)
            preview_items.append((img_path, f"Score: {score:.3f} - {char_dir_name}"))
            
            if len(preview_items) >= max_preview:
                break
        
        if len(preview_items) >= max_preview:
            break
    
    return preview_items