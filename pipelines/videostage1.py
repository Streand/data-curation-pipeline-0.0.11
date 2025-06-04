import os
import logging
import cv2
import numpy as np
import time
import json
from datetime import datetime
from insightface.app import FaceAnalysis  # Import only what we need
import sys
from typing import List, Dict  # Removed unused type imports
from concurrent.futures import ThreadPoolExecutor  # Removed unused as_completed

# Simple logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.device import get_device

# Constants for face detection and frame selection
MIN_FACE_SIZE = 100  # Minimum face size in pixels (width or height)
FACE_CONFIDENCE_THRESHOLD = 0.90  # Minimum confidence for face detection
SHARPNESS_THRESHOLD = 100  # Laplacian variance threshold for sharpness

class VideoProcessor:
    """
    Process videos to extract high-quality frames with faces
    using OpenCV and InsightFace.
    """
    
    def __init__(self, output_dir: str, use_gpu: bool = True) -> None:
        """
        Initialize the VideoProcessor.
        
        Args:
            output_dir: Directory to save extracted frames
            use_gpu: Whether to use GPU acceleration if available
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Use GPU if available and requested
        self.device = get_device() if use_gpu else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize face detection model
        self.face_analyzer = self._initialize_face_detector()
        
        # Set up thread pool for parallel processing when needed
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Store current thresholds (these will be updated from process_batch)
        self.face_confidence_threshold = FACE_CONFIDENCE_THRESHOLD
        self.sharpness_threshold = SHARPNESS_THRESHOLD

    def _initialize_face_detector(self) -> FaceAnalysis:
        """Initialize and configure the InsightFace detector"""
        try:
            # Initialize the face analyzer with appropriate device
            face_app = FaceAnalysis(name="buffalo_l", providers=[
                'CUDAExecutionProvider' if self.device == 'cuda' else 'CPUExecutionProvider'
            ])
            face_app.prepare(ctx_id=0, det_size=(1024, 1024))
            return face_app
        except Exception as e:
            print(f"Error initializing face detector: {e}")
            raise

    def compute_sharpness(self, frame: np.ndarray) -> float:
        """
        Compute image sharpness using Laplacian variance.
        Higher value = sharper image.
        
        Args:
            frame: Input frame as numpy array
        
        Returns:
            Sharpness score
        """
        if frame is None or frame.size == 0:
            return 0.0
            
        # Convert to grayscale if needed
        if len(frame.shape) > 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Compute variance of Laplacian
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return lap_var

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in a frame using InsightFace.
        
        Args:
            frame: Input frame as numpy array
        
        Returns:
            List of face information dictionaries
        """
        if frame is None or frame.size == 0:
            return []
            
        try:
            faces = self.face_analyzer.get(frame)
            face_info = []
            
            for face in faces:
                bbox = face.bbox.astype(int)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Skip if face is too small
                if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
                    continue
                    
                # Extract relevant metrics
                face_dict = {
                    "bbox": bbox.tolist(),
                    "score": float(face.det_score),
                    "size": max(width, height),
                    "landmarks": face.landmark_2d_106.tolist() if hasattr(face, 'landmark_2d_106') else None
                }
                face_info.append(face_dict)
                
            return face_info
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []

    def extract_frames(self, video_path: str, frame_interval: int = 30) -> List[Dict]:
        """
        Extract frames from a video file at specified intervals.
        
        Args:
            video_path: Path to the video file
            frame_interval: Extract every nth frame
        
        Returns:
            List of frame information dictionaries
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = os.path.join(self.output_dir, video_name)
        
        try:
            os.makedirs(frame_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating frame directory {frame_dir}: {e}")
            return []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path}")
        print(f"  - FPS: {fps:.2f}, Total frames: {total_frames}")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - Extracting every {frame_interval} frames")
        
        frames_info = []
        frame_count = 0
        extracted_count = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Process this frame
                sharpness = self.compute_sharpness(frame)
                faces = self.detect_faces(frame)
                
                # Get highest face score if faces detected
                max_face_score = max([face["score"] for face in faces], default=0.0)
                
                # Save basic frame info
                frame_info = {
                    "frame_num": frame_count,
                    "sharpness": sharpness,
                    "faces": faces,
                    "face_count": len(faces),
                    "max_face_score": max_face_score
                }
                
                # Use the instance thresholds which are set by process_batch
                # Save quality frames with faces that pass our threshold
                if faces and max_face_score >= self.face_confidence_threshold and sharpness >= self.sharpness_threshold:
                    frame_filename = f"{video_name}_{frame_count:06d}.jpg"
                    frame_path = os.path.join(frame_dir, frame_filename)
                    
                    try:
                        cv2.imwrite(frame_path, frame)
                        frame_info["path"] = frame_path
                        extracted_count += 1
                    except Exception as e:
                        print(f"Error saving frame to {frame_path}: {e}")
                
                frames_info.append(frame_info)
                
            frame_count += 1
            
            # Show progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%) " 
                      f"- {fps_processing:.1f} FPS - Found {extracted_count} good frames")
        
        cap.release()
        
        print(f"Finished processing {video_path}")
        print(f"  - Processed {frame_count} frames, extracted {extracted_count} frames")
        print(f"  - Elapsed time: {time.time() - start_time:.2f} seconds")
        
        return frames_info

    def select_best_frames(self, 
                           frames_info: List[Dict], 
                           max_frames: int = 10, 
                           min_distance: int = 30,
                           extract_all_good_frames: bool = False) -> List[Dict]:
        """
        Select frames based on face confidence and sharpness.
        
        Args:
            frames_info: List of frame information dictionaries
            max_frames: Maximum number of frames to select (if extract_all_good_frames is False)
            min_distance: Minimum frame distance between selected frames (if extract_all_good_frames is False)
            extract_all_good_frames: If True, return all valid frames without max limit or distance filtering
        
        Returns:
            List of selected frames
        """
        # Filter frames that have faces and a path (saved frames)
        valid_frames = [frame for frame in frames_info if frame.get("face_count", 0) > 0 and "path" in frame]
        
        if not valid_frames:
            return []
        
        # If we want all good frames, just calculate scores and return all
        if extract_all_good_frames:
            # Calculate scores for all valid frames
            for frame in valid_frames:
                frame["score"] = frame.get("max_face_score", 0) * 0.7 + frame.get("sharpness", 0) / 500.0 * 0.3
            
            # Sort by frame number for consistent ordering
            valid_frames.sort(key=lambda x: x.get("frame_num", 0))
            return valid_frames
                
        # Original behavior for selecting best frames with spacing and limits
        scored_frames = [(
            frame,
            frame.get("max_face_score", 0) * 0.7 + frame.get("sharpness", 0) / 500.0 * 0.3
        ) for frame in valid_frames]
        
        scored_frames.sort(key=lambda x: x[1], reverse=True)
        
        # Select frames with minimum distance between them
        selected_frames = []
        selected_frame_nums = []
        
        for frame, score in scored_frames:
            frame_num = frame["frame_num"]
            
            # Check if this frame is far enough from already selected frames
            if all(abs(frame_num - selected) >= min_distance for selected in selected_frame_nums):
                frame_copy = frame.copy()
                frame_copy["score"] = score
                selected_frames.append(frame_copy)
                selected_frame_nums.append(frame_num)
                
                if len(selected_frames) >= max_frames:
                    break
                    
        return selected_frames

    def process_video(self, 
                      video_path: str, 
                      frame_interval: int = 30, 
                      max_frames: int = 10, 
                      min_distance: int = 30,
                      extract_all_good_frames: bool = False) -> Dict:
        """
        Process a video to extract frames.
        
        Args:
            video_path: Path to the video file
            frame_interval: Extract every nth frame
            max_frames: Maximum number of frames to return (if not extract_all_good_frames)
            min_distance: Minimum frame distance between selected frames
            extract_all_good_frames: If True, extract all frames that pass quality thresholds
            
        Returns:
            Dictionary with processing results
        """
        video_name = os.path.basename(video_path)
        start_time = time.time()
        
        try:
            # Extract frames from video
            extracted_frames = self.extract_frames(video_path, frame_interval)
            
            # Select frames (either all good ones or just the best ones)
            best_frames = self.select_best_frames(
                extracted_frames, 
                max_frames=max_frames, 
                min_distance=min_distance,
                extract_all_good_frames=extract_all_good_frames
            )
            
            # Prepare result
            result = {
                "video": video_name,
                "frames_processed": len(extracted_frames),
                "best_frames": best_frames,
                "processing_time": time.time() - start_time,
                "output_dir": self.output_dir,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save metadata to JSON
            metadata_path = os.path.join(self.output_dir, f"{os.path.splitext(video_name)[0]}_metadata.json")
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(result, f, indent=4, cls=NumpyEncoder)
            except Exception as e:
                print(f"Error saving metadata to {metadata_path}: {e}")
                
            return result
            
        except Exception as e:
            error_result = {
                "video": video_name,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"Error processing video {video_name}: {e}")
            return error_result


def process_batch(video_dir: str, 
                 output_dir: str = None, 
                 frame_interval: int = 30,
                 max_frames: int = 10,
                 min_distance: int = 30,
                 face_confidence_threshold: float = FACE_CONFIDENCE_THRESHOLD,
                 sharpness_threshold: float = SHARPNESS_THRESHOLD,
                 extract_all_good_frames: bool = False) -> Dict:
    """
    Process a batch of videos in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted frames (if None, uses video_dir/frames)
        frame_interval: Extract every nth frame
        max_frames: Maximum number of frames to return per video (if not extract_all_good_frames)
        min_distance: Minimum frame distance between selected frames
        face_confidence_threshold: Minimum confidence for face detection
        sharpness_threshold: Minimum sharpness threshold
        extract_all_good_frames: If True, extract all good frames without applying max_frames limit
        
    Returns:
        Dictionary with processing results for all videos
    """
    if not os.path.exists(video_dir):
        return {"error": f"Directory not found: {video_dir}", "total_videos": 0, "results": []}
        
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(video_dir, "frames")
    
    try:    
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        return {"error": f"Error creating output directory: {str(e)}", "total_videos": 0, "results": []}
    
    # Get video files
    video_files = [f for f in os.listdir(video_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                   
    if not video_files:
        return {"error": f"No video files found in {video_dir}", "total_videos": 0, "results": []}
    
    # Initialize processor
    processor = VideoProcessor(output_dir=output_dir)
    
    # Update thresholds on the processor instance
    processor.face_confidence_threshold = face_confidence_threshold
    processor.sharpness_threshold = sharpness_threshold
    
    # Process each video
    results = []
    total_videos = len(video_files)
    
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(video_dir, video_file)
        
        print(f"Processing video {video_file} ({i + 1}/{len(video_files)})")
        
        result = processor.process_video(
            video_path, 
            frame_interval=frame_interval,
            max_frames=max_frames,
            min_distance=min_distance,
            extract_all_good_frames=extract_all_good_frames
        )
        
        results.append(result)
    
    # Prepare batch result
    batch_result = {
        "total_videos": len(video_files),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save batch metadata
    try:
        batch_metadata_path = os.path.join(output_dir, f"batch_metadata_{int(time.time())}.json")
        with open(batch_metadata_path, 'w') as f:
            json.dump(batch_result, f, indent=4, cls=NumpyEncoder)
    except Exception as e:
        print(f"Error saving batch metadata: {e}")
        batch_result["warning"] = f"Failed to save batch metadata: {str(e)}"
        
    return batch_result

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)