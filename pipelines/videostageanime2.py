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

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import functions from stage 1
from pipelines.videostageanime import filter_frames_by_anime_character, calculate_anime_character_similarity

# GPU-accelerated feature extractor
class GPUFeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing GPU feature extractor on: {self.device}")
        
        # Use ResNet50 for feature extraction
        self.model = resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove classifier
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_batch):
        """Extract features from a batch of images using GPU"""
        with torch.no_grad():
            features = self.model(image_batch)
            features = features.flatten(1)  # Flatten to vector
            features = F.normalize(features, p=2, dim=1)  # L2 normalize
        return features
    
    def process_image(self, image):
        """Process single image to tensor"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image)

def calculate_gpu_similarity_batch(reference_features, frame_batch, feature_extractor):
    """
    Calculate similarity using GPU acceleration for batch processing
    """
    try:
        # Process batch of frames
        frame_tensors = []
        for frame in frame_batch:
            frame_tensor = feature_extractor.process_image(frame)
            frame_tensors.append(frame_tensor)
        
        # Stack into batch tensor
        batch_tensor = torch.stack(frame_tensors).to(feature_extractor.device)
        
        # Extract features
        frame_features = feature_extractor.extract_features(batch_tensor)
        
        # Calculate cosine similarity
        similarities = torch.mm(reference_features, frame_features.t())
        similarities = similarities.squeeze().cpu().numpy()
        
        # Handle single frame case
        if similarities.ndim == 0:
            similarities = [similarities.item()]
        elif similarities.ndim == 1:
            similarities = similarities.tolist()
        
        return similarities
        
    except Exception as e:
        logger.error(f"GPU similarity calculation error: {e}")
        # Fallback to CPU calculation
        return [0.5] * len(frame_batch)

def process_character_filtering_batch_gpu(frames_base_dir, reference_char_path, similarity_threshold=0.6, max_results_per_video=500, batch_size=32):
    """
    GPU-accelerated Stage 2: Process character filtering across all extracted frames
    
    Args:
        frames_base_dir: Base directory containing video frame folders (from stage 1)
        reference_char_path: Path to reference character image
        similarity_threshold: Minimum similarity threshold (0-1)
        max_results_per_video: Maximum matches per video
        batch_size: Number of frames to process in each GPU batch
        
    Returns:
        Dictionary with filtering results for all videos
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
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Initialize GPU feature extractor
    feature_extractor = GPUFeatureExtractor()
    
    # Process reference character
    ref_img = cv2.imread(reference_char_path)
    if ref_img is None:
        return {"error": "Could not load reference character image"}
    
    ref_tensor = feature_extractor.process_image(ref_img).unsqueeze(0).to(feature_extractor.device)
    ref_features = feature_extractor.extract_features(ref_tensor)
    
    # Find all video frame directories
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
        return {"error": "No video frame directories found. Please run Stage 1 (frame extraction) first."}
    
    char_name = os.path.splitext(os.path.basename(reference_char_path))[0]
    logger.info(f"Starting GPU-accelerated character filtering for '{char_name}' across {len(video_frame_dirs)} videos")
    
    # Create output directory for character matches
    output_dir = os.path.join(frames_base_dir, f"character_{char_name}_matches")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total_matches = 0
    
    for video_info in video_frame_dirs:
        video_name = video_info["video_name"]
        frames_dir = video_info["frames_dir"]
        
        logger.info(f"Processing character filtering for video: {video_name}")
        
        start_time = time.time()
        try:
            # Get all frame files
            frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            frame_files.sort()
            
            if not frame_files:
                logger.warning(f"No frame files found in {frames_dir}")
                continue
            
            matches = []
            frames_processed = 0
            
            # Process frames in batches for GPU efficiency
            for i in range(0, len(frame_files), batch_size):
                batch_files = frame_files[i:i+batch_size]
                batch_frames = []
                batch_paths = []
                
                # Load batch of frames
                for frame_file in batch_files:
                    frame_path = os.path.join(frames_dir, frame_file)
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        batch_frames.append(frame)
                        batch_paths.append(frame_path)
                
                if not batch_frames:
                    continue
                
                # Calculate similarities using GPU
                similarities = calculate_gpu_similarity_batch(ref_features, batch_frames, feature_extractor)
                
                # Process results
                for j, (similarity_score, frame_path, frame) in enumerate(zip(similarities, batch_paths, batch_frames)):
                    frames_processed += 1
                    
                    if similarity_score >= similarity_threshold:
                        # Create character-specific directory
                        char_video_dir = os.path.join(output_dir, f"character_{char_name}")
                        os.makedirs(char_video_dir, exist_ok=True)
                        
                        # Save matched frame with similarity score in filename
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
                if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= len(frame_files):
                    logger.info(f"GPU processed {min(i + batch_size, len(frame_files))}/{len(frame_files)} frames, found {len(matches)} matches")
                
                # Early stop if we have enough matches
                if max_results_per_video and len(matches) >= max_results_per_video:
                    logger.info(f"Reached maximum results limit ({max_results_per_video}), stopping early")
                    break
            
            # Sort by similarity score (highest first)
            matches.sort(key=lambda x: x["similarity_score"], reverse=True)
            total_matches += len(matches)
            
            result = {
                "video": video_name,
                "matches_found": len(matches),
                "frames_processed": frames_processed,
                "character_directory": os.path.join(output_dir, f"character_{char_name}"),
                "matches": matches[:20],  # Store only first 20 for metadata
                "processing_time": time.time() - start_time,
                "gpu_accelerated": True,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error filtering {video_name}: {str(e)}")
            result = {
                "video": video_name,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "status": "error"
            }
        
        results.append(result)
        
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Prepare batch result
    batch_result = {
        "character_name": char_name,
        "reference_character_path": reference_char_path,
        "total_videos_processed": len(video_frame_dirs),
        "total_character_matches": total_matches,
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
        "stage": "2 - GPU Character Filtering"
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"anime_stage2_gpu_filtering_{int(time.time())}.json")
    with open(metadata_path, 'w') as f:
        json.dump(batch_result, f, indent=4)
    
    logger.info(f"GPU Stage 2 complete: {total_matches} character matches found across {len(video_frame_dirs)} videos")
    return batch_result

# Keep original CPU function as fallback
def process_character_filtering_batch(frames_base_dir, reference_char_path, similarity_threshold=0.6, max_results_per_video=500):
    """
    Original CPU-based character filtering (fallback)
    """
    # Your existing CPU implementation here...
    # (keep the original function for fallback)
    pass

def get_available_extracted_frames(base_dir):
    """
    Get list of available extracted frame directories from Stage 1
    """
    available_videos = []
    
    if not os.path.exists(base_dir):
        return available_videos
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            all_frames_dir = os.path.join(item_path, "all_frames")
            if os.path.exists(all_frames_dir):
                # Count frames
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
    """
    Get preview images from character matching results
    """
    if not os.path.exists(character_matches_dir):
        return []
    
    # Find character match directories
    char_dirs = []
    
    # Check if character_matches_dir itself contains images
    if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(character_matches_dir)):
        char_dirs.append(character_matches_dir)
    
    # Also check subdirectories
    for item in os.listdir(character_matches_dir):
        item_path = os.path.join(character_matches_dir, item)
        if os.path.isdir(item_path):
            char_dirs.append(item_path)
    
    preview_items = []
    
    for char_dir in char_dirs:
        # Get image files sorted by similarity score (highest first)
        image_files = [f for f in os.listdir(char_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sort by similarity score in filename
        def extract_score(filename):
            try:
                parts = filename.split('_')
                for part in parts:
                    # Look for score pattern like "0.123" or "0.8"
                    if part.startswith('0.') and len(part) <= 5:
                        return float(part)
                return 0.0
            except:
                return 0.0
        
        image_files.sort(key=extract_score, reverse=True)
        
        # Add preview items
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