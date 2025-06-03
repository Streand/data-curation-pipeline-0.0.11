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
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Default thresholds for stage 1
FACE_CONFIDENCE_THRESHOLD = 0.7
SHARPNESS_THRESHOLD = 40.0

def check_and_initialize_gpu():
    """Initialize GPU properly and report status"""
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device)
            memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            
            # Check for Blackwell architecture
            is_blackwell = "50" in device_name or "blackwell" in device_name.lower()
            cuda_version = torch.version.cuda
            
            print(f"GPU initialized: {device_name}")
            print(f"CUDA Version: {cuda_version}")
            print(f"Total VRAM: {memory_total:.2f} GB")
            
            if is_blackwell:
                print("Detected Blackwell architecture GPU")
                print("Using optimized settings for Blackwell")
                
                # For Blackwell GPUs, don't set memory fraction as it's deprecated
                # Instead use more modern approach if needed
                if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                    print("CUDA AMP autocast available (recommended for Blackwell)")
            else:
                # Legacy approach for older GPUs
                torch_version = torch.__version__
                if int(torch_version.split('.')[0]) >= 2:
                    # Modern PyTorch approach
                    print("Using modern PyTorch memory management")
                    # Let PyTorch handle memory allocation dynamically
                else:
                    # Legacy approach
                    print("Using legacy memory management")
                    torch.cuda.set_per_process_memory_fraction(0.8)
                    
            # Test basic tensor operations
            try:
                x = torch.randn(100, 100, device='cuda')
                y = x @ x.t()
                del x, y
                torch.cuda.empty_cache()
                print("Basic CUDA tensor operations: SUCCESS")
            except Exception as e:
                print(f"WARNING: CUDA tensor test failed: {e}")
                
            return True
        except Exception as e:
            print(f"ERROR initializing GPU: {e}")
            return False
    else:
        print("CUDA not available - using CPU")
        return False

HAS_GPU = check_and_initialize_gpu()

def verify_face_detection_gpu_usage(img):
    """Test face detection to verify it's using GPU"""
    print("Testing GPU usage in face detection...")
    
    import time
    
    if torch.cuda.is_available():
        try:
            # Record time for CPU detection
            start_cpu = time.time()
            # Force CPU detection first
            torch.cuda.empty_cache()  # Clear GPU memory first
            with torch.no_grad():
                with torch.cpu.device("cpu"):
                    faces_cpu = detect_faces(img)
            cpu_time = time.time() - start_cpu
            
            # Record time for GPU detection
            torch.cuda.empty_cache()  # Clear GPU memory first
            start_gpu = time.time()
            faces_gpu = detect_faces(img)
            gpu_time = time.time() - start_gpu
            
            # Calculate speedup
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"Face detection: CPU {cpu_time:.3f}s, GPU {gpu_time:.3f}s, {speedup:.1f}x speedup")
                
                if speedup < 1.5:
                    print("WARNING: GPU acceleration not working properly")
            
            print(f"Found {len(faces_gpu)} faces using GPU")
            return faces_gpu
        except Exception as e:
            print(f"ERROR in GPU test: {str(e)}")
            print("Continuing with CPU...")
            return detect_faces(img)
    else:
        print("No GPU available, using CPU")
        return detect_faces(img)

def get_app_root():
    """Get the application root directory in a portable way"""
    current_file = os.path.abspath(__file__)
    pipelines_dir = os.path.dirname(current_file)
    app_root = os.path.dirname(pipelines_dir)
    return app_root

def extract_frames(video_path, output_dir, sample_rate=1):
    """Extract frames from video at given sample rate"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
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

def extract_frames_optimized(video_path, output_dir, sample_rate=1, progress=None, check_stop=None):
    """Thread-safe frame extraction"""
    # Safeguard against bool check_stop
    is_stop_requested = lambda: False
    if callable(check_stop):
        is_stop_requested = check_stop
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(sample_rate)
    video.release()
    
    def process_frame(frame_idx):
        try:
            # IMPORTANT: Replace all direct calls to check_stop() with is_stop_requested()
            if is_stop_requested():
                return None
        except:
            pass
        
        thread_video = cv2.VideoCapture(video_path)
        if not thread_video.isOpened():
            return None
            
        thread_video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = thread_video.read()
        thread_video.release()
        
        if not success:
            return None
            
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        return frame_path
    
    frames_to_process = list(range(0, frame_count, frame_interval))
    max_workers = min(os.cpu_count() or 4, 4)
    
    saved_frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_frame, idx) for idx in frames_to_process]
        
        for i, future in enumerate(as_completed(futures)):
            if progress:
                progress(i / len(frames_to_process), f"Extracting frames: {i}/{len(frames_to_process)}")
                
            # Always use is_stop_requested instead of check_stop
            if is_stop_requested():
                break
                
            frame_path = future.result()
            if frame_path:
                saved_frames.append(frame_path)
    
    return saved_frames, fps

def check_gpu_usage():
    """Check if GPU is being used and how much memory is allocated"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)
        
        print(f"Using: {device_name}")
        print(f"Memory allocated: {memory_allocated:.2f} MB")
        print(f"Memory reserved: {memory_reserved:.2f} MB")
        
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
    """First-stage fast filtering with permissive thresholds to maximize recall"""
    device = get_device()
    results = []
    
    if weights is None:
        weights = {
            "face_confidence": 0.6,
            "sharpness": 0.3,
            "face_size": 0.1,
        }
    
    if thresholds is None:
        thresholds = {
            "face_confidence": FACE_CONFIDENCE_THRESHOLD,
            "sharpness": SHARPNESS_THRESHOLD
        }
    
    total_frames = len(frame_paths)
    
    for i, frame_path in enumerate(frame_paths):
        try:
            if progress is not None:
                progress(i/total_frames, f"Fast analysis: {i}/{total_frames} frames")
            
            img = cv2.imread(frame_path)
            if img is None:
                raise Exception(f"Could not load image: {frame_path}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            faces = detect_faces(img)
            
            if not faces:
                results.append({
                    "path": frame_path,
                    "score": sharpness / 100,
                    "sharpness": sharpness,
                    "face_score": 0,
                    "face_size": 0,
                    "faces": 0,
                    "has_face": False,
                    "passed_threshold": sharpness >= thresholds["sharpness"]
                })
                continue
            
            max_face_size = 0
            max_face_score = 0
            for face in faces:
                face_size = face["bbox"][2] * face["bbox"][3]
                if face_size > max_face_size:
                    max_face_size = face_size
                    max_face_score = face["score"]
            
            passed_threshold = (max_face_score >= thresholds["face_confidence"] or 
                               sharpness >= thresholds["sharpness"])
            
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
                "stage": 1
            })
            
        except Exception as e:
            results.append({
                "path": frame_path,
                "score": 0,
                "error": str(e),
                "passed_threshold": False,
                "stage": 1
            })
    
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results

def score_frames_batched(frame_paths, batch_size=8, weights=None, thresholds=None, progress=None, check_stop=None):
    """Process frames in batches for better GPU utilization"""
    # Safeguard against bool check_stop
    is_stop_requested = lambda: False
    if callable(check_stop):
        is_stop_requested = check_stop
        
    device = get_device()
    results = []
    
    if weights is None:
        weights = {
            "face_confidence": 0.6,
            "sharpness": 0.3,
            "face_size": 0.1,
        }
    
    if thresholds is None:
        thresholds = {
            "face_confidence": FACE_CONFIDENCE_THRESHOLD,
            "sharpness": SHARPNESS_THRESHOLD
        }
    
    # Dynamically adjust batch size based on GPU memory if available
    if torch.cuda.is_available():
        try:
            # Get available memory and adjust batch size
            free_memory = torch.cuda.mem_get_info()[0] / (1024**3)  # Free memory in GB
            device_name = torch.cuda.get_device_name()
            
            # Adjust batch size based on available memory - higher for Blackwell
            if "50" in device_name or "blackwell" in device_name.lower():
                # More aggressive for Blackwell architecture
                batch_size = max(8, min(32, int(free_memory * 4)))
                print(f"Using optimized batch size for Blackwell: {batch_size}")
            else:
                # Conservative for other architectures
                batch_size = max(4, min(16, int(free_memory * 2)))
                print(f"Using batch size: {batch_size}")
        except Exception as e:
            print(f"Error adjusting batch size: {e}")
            # Keep original batch size
    
    total_frames = len(frame_paths)
    batches = [frame_paths[i:i+batch_size] for i in range(0, total_frames, batch_size)]
    
    processed_count = 0
    
    # Create CUDA streams for parallel processing if available
    streams = []
    if torch.cuda.is_available():
        try:
            # Create streams for parallel processing
            for _ in range(min(4, batch_size)):
                streams.append(torch.cuda.Stream())
        except Exception as e:
            print(f"Warning: Could not create CUDA streams: {e}")
    
    for batch_idx, batch in enumerate(batches):
        try:
            if is_stop_requested():
                break
                
            if progress:
                progress(processed_count/total_frames, f"Analyzing frames: {processed_count}/{total_frames}")
            
            # Rest of your batch processing code...
            with ThreadPoolExecutor(max_workers=batch_size) as loader:
                image_futures = {loader.submit(cv2.imread, path): path for path in batch}
                batch_images = {}
                
                for future in as_completed(image_futures):
                    path = image_futures[future]
                    try:
                        img = future.result()
                        if img is not None:
                            batch_images[path] = img
                    except Exception:
                        pass
            
            # Use CUDA streams if available for parallel processing
            if streams and torch.cuda.is_available():
                try:
                    with torch.cuda.stream(streams[batch_idx % len(streams)]):
                        # Process face detection in this stream
                        paths = list(batch_images.keys())
                        images = list(batch_images.values())
                        
                        # Calculate sharpness and get face results
                        sharpness_values = []
                        for img in images:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                            sharpness_values.append(sharpness)
                        
                        face_results = []
                        for img in images:
                            face_results.append(detect_faces(img))
                except Exception as e:
                    print(f"Error in CUDA stream processing: {e}")
                    # Fall back to standard processing
                    paths = list(batch_images.keys())
                    images = list(batch_images.values())
                    
                    sharpness_values = []
                    for img in images:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                        sharpness_values.append(sharpness)
                    
                    face_results = []
                    for img in images:
                        face_results.append(detect_faces(img))
            else:
                # Original non-stream processing
                paths = list(batch_images.keys())
                images = list(batch_images.values())
                
                sharpness_values = []
                for img in images:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    sharpness_values.append(sharpness)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                face_results = []
                for img in images:
                    face_results.append(detect_faces(img))
            
            # Rest of the result processing remains the same
            for i, (path, img, sharpness) in enumerate(zip(paths, images, sharpness_values)):
                faces = face_results[i]
                
                if not faces:
                    results.append({
                        "path": path,
                        "score": sharpness / 100,
                        "sharpness": sharpness,
                        "face_score": 0,
                        "face_size": 0,
                        "faces": 0,
                        "has_face": False,
                        "passed_threshold": sharpness >= thresholds["sharpness"]
                    })
                    continue
                
                max_face_size = 0
                max_face_score = 0
                for face in faces:
                    face_size = face["bbox"][2] * face["bbox"][3]
                    if face_size > max_face_size:
                        max_face_size = face_size
                        max_face_score = face["score"]
                
                passed_threshold = (max_face_score >= thresholds["face_confidence"] or 
                                  sharpness >= thresholds["sharpness"])
                
                overall_score = (
                    weights["face_confidence"] * max_face_score +
                    weights["sharpness"] * (sharpness / 100) +
                    weights["face_size"] * (max_face_size / 10000)
                )
                
                results.append({
                    "path": path,
                    "score": overall_score,
                    "sharpness": sharpness,
                    "face_score": max_face_score,
                    "face_size": max_face_size,
                    "faces": len(faces),
                    "has_face": True,
                    "passed_threshold": passed_threshold,
                    "stage": 1
                })
                
            processed_count += len(batch)
            
        except torch.cuda.OutOfMemoryError:
            # Handle CUDA OOM specifically
            print("CUDA out of memory error - reducing batch size and retrying")
            torch.cuda.empty_cache()
            
            # Cut batch size and retry this batch
            new_batch_size = max(1, batch_size // 2)
            sub_batches = [batch[i:i+new_batch_size] for i in range(0, len(batch), new_batch_size)]
            
            for sub_batch in sub_batches:
                # Process smaller batch (simplified - you might want to refactor to avoid code duplication)
                sub_results = score_frames_batched(
                    sub_batch, 
                    batch_size=new_batch_size,
                    weights=weights,
                    thresholds=thresholds
                )
                results.extend(sub_results)
                processed_count += len(sub_batch)
                
                if progress:
                    progress(processed_count/total_frames, f"Analyzing frames: {processed_count}/{total_frames}")
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # Continue with next batch
    
    # Clean up CUDA resources
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
            
    return results

def select_frames_stage1(video_path, output_dir=None, preset="TikTok/Instagram", 
                      sample_rate=None, num_frames=None, min_frame_distance=30, 
                      use_scene_detection=False, progress=None, check_stop=None,
                      batch_size=8, motion_threshold=25):
    """Fast extraction of potential frames from video"""
    
    # Always create a safe version of check_stop
    # This lambda function will never raise an error even if check_stop is None, a bool, or anything else
    is_stop_requested = lambda: False
    
    # Only use check_stop if it's actually a callable function
    if check_stop is not None and callable(check_stop):
        try:
            # Test that it works
            _ = check_stop()
            # If we get here, check_stop is working correctly
            is_stop_requested = check_stop
        except Exception as e:
            print(f"Warning: check_stop function failed: {e}. Using default.")
    
    # Now use is_stop_requested everywhere in this function and pass it to other functions

    temp_video = cv2.VideoCapture(video_path)
    success, first_frame = temp_video.read()
    temp_video.release()
    
    app_root = get_app_root()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    storage_dir = os.path.join(app_root, "store_images", "video_stage_1", f"{video_name}_{timestamp}")
    os.makedirs(storage_dir, exist_ok=True)
    output_dir = storage_dir
    
    preset_config = VIDEO_PRESETS[preset]
    
    sample_rate = sample_rate if sample_rate is not None else preset_config["sample_rate"]
    num_frames = num_frames if num_frames is not None else preset_config["number_of_best_frames"]
    min_frame_distance = min_frame_distance if min_frame_distance is not None else preset_config["thresholds"]["minimum_frame_distance"]
    
    weights = preset_config["scoring_weights"]
    thresholds = {
        "face_confidence": preset_config["thresholds"]["face_confidence"],  
        "sharpness": SHARPNESS_THRESHOLD
    }
    
    frames, fps = extract_frames_optimized(
        video_path, 
        output_dir, 
        sample_rate=sample_rate,
        progress=progress,
        check_stop=is_stop_requested  # Pass the safer function
    )
    
    if not frames:
        print("No frames were extracted from the video.")
        return [], fps, 0, None, None
    
    scored_frames = score_frames_batched(
        frames, 
        batch_size=batch_size,
        weights=weights, 
        thresholds=thresholds, 
        progress=progress,
        check_stop=is_stop_requested  # Pass the safer function
    )
    
    passed_frames = [f for f in scored_frames if f.get("passed_threshold", False)]
    
    if len(passed_frames) < min(5, num_frames):
        print(f"Only {len(passed_frames)} frames passed filters, adding more frames...")
        additional_needed = min(5, num_frames) - len(passed_frames)
        
        failed_frames = [f for f in scored_frames if not f.get("passed_threshold", False)]
        failed_frames.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        for i in range(min(additional_needed, len(failed_frames))):
            failed_frames[i]["passed_threshold"] = True
            passed_frames.append(failed_frames[i])
    
    # Add these safety checks before processing frames
    if len(frames) == 0:
        print(f"No frames extracted from video: {video_path}")
        return [], fps, 0, None, None

    if len(passed_frames) == 0:
        print(f"No frames passed quality thresholds in video: {video_path}")
        return [], fps, 0, None, None

    # Create a safer version of the frame selection code
    print(f"Selecting diverse frames from {len(passed_frames)} candidates...")
    diverse_frames = []
    selected_indices = []
    
    # Keep track of actual frame paths for debugging
    frame_paths_only = [frame for frame in frames]
    
    # Create safer lookup with both normalized and original paths
    frame_paths_map = {}
    for i, path in enumerate(frame_paths_only):
        try:
            norm_path = os.path.normpath(os.path.abspath(path))
            frame_paths_map[norm_path] = i
            frame_paths_map[path] = i  # Also add original path
        except:
            pass
    
    # Sort passed frames by score for better selection
    passed_frames.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    for frame in passed_frames:
        try:
            # Get the frame path and try multiple ways to find its index
            frame_path = frame.get("path", "")
            if not frame_path:
                continue
                
            frame_idx = None
            
            # Try direct lookup in map
            if frame_path in frame_paths_map:
                frame_idx = frame_paths_map[frame_path]
            else:
                # Try normalized path
                try:
                    norm_path = os.path.normpath(os.path.abspath(frame_path))
                    if norm_path in frame_paths_map:
                        frame_idx = frame_paths_map[norm_path]
                except:
                    pass
            
            # If still not found, try basename match as last resort
            if frame_idx is None:
                try:
                    basename = os.path.basename(frame_path)
                    for i, path in enumerate(frame_paths_only):
                        if os.path.basename(path) == basename:
                            frame_idx = i
                            break
                except:
                    pass
                    
            # If we found a valid index, check frame distance
            if frame_idx is not None:
                if all(abs(frame_idx - selected) >= min_frame_distance for selected in selected_indices):
                    diverse_frames.append(frame)
                    selected_indices.append(frame_idx)
                    
                if len(diverse_frames) >= num_frames:
                    break
            else:
                print(f"Warning: Could not locate index for frame: {frame_path}")
                
        except Exception as e:
            print(f"Error during frame selection: {str(e)}")
            continue
    
    # Rest of function remains the same...
    print(f"Stage 1 complete:")
    print(f"- Total frames extracted: {len(frames)}")
    print(f"- Frames passing thresholds: {len(passed_frames)}")
    print(f"- Diverse frames selected: {len(diverse_frames)}")
    
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
    
    best_frames_dir = os.path.join(
        app_root, 
        "store_images", 
        "video_stage_1_best", 
        f"{video_name}_{timestamp}"
    )
    os.makedirs(best_frames_dir, exist_ok=True)

    for frame in diverse_frames:
        try:
            src_path = frame["path"]
            filename = os.path.basename(src_path)
            score_str = f"{frame.get('score', 0):.3f}".replace(".", "_")
            new_filename = f"score_{score_str}_{filename}"
            dst_path = os.path.join(best_frames_dir, new_filename)
            
            shutil.copy2(src_path, dst_path)
            frame["best_path"] = dst_path
        except Exception as e:
            print(f"Error copying best frame: {e}")

    with open(os.path.join(best_frames_dir, "best_frames_info.json"), 'w') as f:
        json.dump({
            "video_path": video_path,
            "fps": fps, 
            "total_frames": len(frames),
            "best_frames": diverse_frames
        }, f, indent=2)

    print(f"Best {len(diverse_frames)} frames copied to: {best_frames_dir}")
    return diverse_frames, fps, len(frames), storage_dir, best_frames_dir