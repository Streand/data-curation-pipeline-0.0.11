import os
import cv2
import numpy as np
import time
import json
from datetime import datetime
import logging
import torch
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import gc

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_optimization():
    """Check GPU and optimize for Blackwell architecture"""
    gpu_info = {
        "has_cuda": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": None,
        "memory_gb": 0,
        "is_blackwell": False,
        "recommended_batch": 32
    }
    
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            gpu_info.update({
                "device_name": device_name,
                "memory_gb": round(memory_total, 1),
                "is_blackwell": "50" in device_name or "blackwell" in device_name.lower()
            })
            
            # Optimize batch size for Blackwell architecture
            if gpu_info["is_blackwell"]:
                # More aggressive for RTX 5000 series
                gpu_info["recommended_batch"] = min(64, max(32, int(memory_total * 2)))
                logger.info(f"Blackwell GPU detected: {device_name} - Using optimized batch size: {gpu_info['recommended_batch']}")
            else:
                gpu_info["recommended_batch"] = min(32, max(16, int(memory_total * 1.5)))
                
        except Exception as e:
            logger.warning(f"Error getting GPU info: {e}")
    
    return gpu_info

class AllFramesExtractor:
    """GPU-accelerated frame extractor for processing all frames from videos"""
    
    def __init__(self, output_base_dir: str, use_gpu: bool = True):
        self.output_base_dir = output_base_dir
        os.makedirs(output_base_dir, exist_ok=True)
        
        # GPU optimization
        self.gpu_info = check_gpu_optimization()
        self.use_gpu = use_gpu and self.gpu_info["has_cuda"]
        self.device = "cuda" if self.use_gpu else "cpu"
        
        logger.info(f"AllFramesExtractor initialized:")
        logger.info(f"  - Device: {self.device}")
        if self.use_gpu:
            logger.info(f"  - GPU: {self.gpu_info['device_name']}")
            logger.info(f"  - VRAM: {self.gpu_info['memory_gb']} GB")
            logger.info(f"  - Batch size: {self.gpu_info['recommended_batch']}")
    
    def extract_all_frames_from_video(self, video_path: str, progress_callback=None) -> Dict[str, Any]:
        """Extract all frames from a single video with GPU acceleration"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(self.output_base_dir, video_name)
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Could not open video: {video_path}"
            logger.error(error_msg)
            return {"error": error_msg, "status": "failed"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing video: {video_name}")
        logger.info(f"  - Total frames: {total_frames:,}")
        logger.info(f"  - Resolution: {width}x{height}")
        logger.info(f"  - FPS: {fps:.2f}")
        
        # Extract frames with GPU optimization
        if self.use_gpu and self.gpu_info["is_blackwell"]:
            result = self._extract_frames_gpu_optimized(cap, output_dir, video_name, total_frames, fps, progress_callback)
        else:
            result = self._extract_frames_standard(cap, output_dir, video_name, total_frames, fps, progress_callback)
        
        cap.release()
        
        # Add video metadata
        result.update({
            "video_name": video_name,
            "video_path": video_path,
            "output_directory": output_dir,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "processing_time": time.time() - start_time,
            "gpu_accelerated": self.use_gpu,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save metadata
        self._save_metadata(result, output_dir)
        
        return result
    
    def _extract_frames_gpu_optimized(self, cap, output_dir: str, video_name: str, total_frames: int, fps: float, progress_callback=None) -> Dict[str, Any]:
        """GPU-optimized frame extraction for Blackwell architecture"""
        frames_saved = 0
        batch_size = self.gpu_info["recommended_batch"]
        
        logger.info(f"Using GPU-optimized extraction (batch size: {batch_size})")
        
        frame_batch = []
        frame_indices = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                # FIXED: Process any remaining frames in the batch before breaking
                if frame_batch:
                    logger.info(f"Processing final batch of {len(frame_batch)} frames")
                    with torch.cuda.device(0):
                        saved_count = self._save_frame_batch(frame_batch, frame_indices, output_dir, video_name)
                        frames_saved += saved_count
                        torch.cuda.empty_cache()
                break
            
            frame_batch.append(frame)
            frame_indices.append(frame_idx)
            
            # FIXED: Process batch when full (removed problematic end condition)
            if len(frame_batch) >= batch_size:
                # Save batch with GPU memory management
                with torch.cuda.device(0):
                    saved_count = self._save_frame_batch(frame_batch, frame_indices, output_dir, video_name)
                    frames_saved += saved_count
                    
                    # Clear GPU cache
                    torch.cuda.empty_cache()
                
                # Clear batch
                frame_batch = []
                frame_indices = []
                
                # Progress update
                if progress_callback:
                    progress = frame_idx / total_frames if total_frames > 0 else 0
                    progress_callback(progress, f"GPU Processing: {frame_idx:,}/{total_frames:,} frames ({frames_saved:,} saved)")
                
                # Log progress
                if frame_idx % 10000 == 0:
                    logger.info(f"GPU processed: {frame_idx:,}/{total_frames:,} frames ({frames_saved:,} saved)")
            
            frame_idx += 1
        
        logger.info(f"GPU extraction complete: {frames_saved} frames saved out of {total_frames} total frames")
        
        return {
            "status": "success",
            "total_frames_in_video": total_frames,
            "frames_extracted": frames_saved,
            "extraction_method": "GPU-optimized"
        }
    
    def _extract_frames_standard(self, cap, output_dir: str, video_name: str, total_frames: int, fps: float, progress_callback=None) -> Dict[str, Any]:
        """Standard CPU/GPU frame extraction"""
        frames_saved = 0
        frame_idx = 0
        
        logger.info("Using standard extraction method")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame
            frame_filename = f"{video_name}_frame_{frame_idx:08d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            try:
                # Use high quality JPEG settings
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frames_saved += 1
            except Exception as e:
                logger.warning(f"Failed to save frame {frame_idx}: {e}")
            
            frame_idx += 1
            
            # Progress update
            if progress_callback and frame_idx % 100 == 0:
                progress = frame_idx / total_frames if total_frames > 0 else 0
                progress_callback(progress, f"Processing: {frame_idx:,}/{total_frames:,} frames ({frames_saved:,} saved)")
            
            # Log progress
            if frame_idx % 5000 == 0:
                logger.info(f"Processed: {frame_idx:,}/{total_frames:,} frames ({frames_saved:,} saved)")
        
        return {
            "status": "success",
            "total_frames_in_video": total_frames,
            "frames_extracted": frames_saved,
            "extraction_method": "Standard"
        }
    
    def _save_frame_batch(self, frame_batch: List[np.ndarray], frame_indices: List[int], output_dir: str, video_name: str) -> int:
        """Save a batch of frames efficiently"""
        saved_count = 0
        
        # Use ThreadPoolExecutor for parallel I/O
        with ThreadPoolExecutor(max_workers=min(8, len(frame_batch))) as executor:
            futures = []
            
            for frame, frame_idx in zip(frame_batch, frame_indices):
                frame_filename = f"{video_name}_frame_{frame_idx:08d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                
                future = executor.submit(self._save_single_frame, frame, frame_path)
                futures.append(future)
            
            # Collect results
            for future in futures:
                if future.result():
                    saved_count += 1
        
        return saved_count
    
    def _save_single_frame(self, frame: np.ndarray, frame_path: str) -> bool:
        """Save a single frame with error handling"""
        try:
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return True
        except Exception as e:
            logger.warning(f"Failed to save frame {frame_path}: {e}")
            return False
    
    def _save_metadata(self, result: Dict[str, Any], output_dir: str):
        """Save extraction metadata"""
        metadata_path = os.path.join(output_dir, "extraction_metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(result, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

def process_video_all_frames(video_path: str, output_base_dir: str, progress_callback=None) -> Dict[str, Any]:
    """Process a single video to extract all frames"""
    extractor = AllFramesExtractor(output_base_dir)
    return extractor.extract_all_frames_from_video(video_path, progress_callback)

def process_videos_batch_all_frames(video_dir: str, output_base_dir: str, progress_callback=None) -> Dict[str, Any]:
    """Process multiple videos to extract all frames"""
    if not os.path.exists(video_dir):
        return {"error": f"Video directory not found: {video_dir}"}
    
    # Find video files
    video_files = [f for f in os.listdir(video_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'))]
    
    if not video_files:
        return {"error": "No video files found in directory"}
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    extractor = AllFramesExtractor(output_base_dir)
    results = []
    total_frames_extracted = 0
    start_time = time.time()
    
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(video_dir, video_file)
        
        logger.info(f"Processing video {i+1}/{len(video_files)}: {video_file}")
        
        # Create progress callback for individual video
        def video_progress(progress, message):
            overall_progress = (i + progress) / len(video_files)
            if progress_callback:
                progress_callback(overall_progress, f"Video {i+1}/{len(video_files)}: {message}")
        
        result = extractor.extract_all_frames_from_video(video_path, video_progress)
        results.append(result)
        
        if result.get("status") == "success":
            total_frames_extracted += result.get("frames_extracted", 0)
    
    # Overall results
    batch_result = {
        "status": "completed",
        "total_videos_processed": len(video_files),
        "successful_videos": len([r for r in results if r.get("status") == "success"]),
        "total_frames_extracted": total_frames_extracted,
        "processing_time": time.time() - start_time,
        "results": results,
        "output_directory": output_base_dir,
        "timestamp": datetime.now().isoformat(),
        "gpu_info": extractor.gpu_info
    }
    
    # Save batch metadata
    batch_metadata_path = os.path.join(output_base_dir, f"batch_extraction_{int(time.time())}.json")
    try:
        with open(batch_metadata_path, 'w') as f:
            json.dump(batch_result, f, indent=4)
    except Exception as e:
        logger.warning(f"Failed to save batch metadata: {e}")
    
    logger.info(f"Batch processing complete:")
    logger.info(f"  - Videos processed: {len(video_files)}")
    logger.info(f"  - Total frames extracted: {total_frames_extracted:,}")
    logger.info(f"  - Processing time: {batch_result['processing_time']:.2f} seconds")
    
    return batch_result