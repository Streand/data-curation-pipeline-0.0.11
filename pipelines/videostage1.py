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
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        
        print(f"GPU initialized: {device_name}")
        print(f"Total VRAM: {memory_total:.2f} GB")
        
        torch.cuda.set_per_process_memory_fraction(0.8)
        return True
    else:
        print("CUDA not available - using CPU")
        return False

HAS_GPU = check_and_initialize_gpu()

def verify_face_detection_gpu_usage(img):
    """Verify that face detection is using GPU effectively"""
    print("Testing GPU usage in face detection...")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        baseline_mem = torch.cuda.memory_allocated() / (1024**2)
        
        faces = detect_faces(img)
        
        torch.cuda.synchronize()
        after_mem = torch.cuda.memory_allocated() / (1024**2)
        diff = after_mem - baseline_mem
        
        print(f"GPU memory used for detection: {diff:.2f} MB")
        if diff < 10:
            print("WARNING: Face detection may not be using GPU properly!")
        else:
            print(f"Found {len(faces)} faces using GPU")
    else:
        print("No GPU available")

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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(sample_rate)
    video.release()
    
    def process_frame(frame_idx):
        if check_stop and check_stop():
            return None
            
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
                
            if check_stop and check_stop():
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
    batches = [frame_paths[i:i+batch_size] for i in range(0, total_frames, batch_size)]
    
    processed_count = 0
    for batch_idx, batch in enumerate(batches):
        if check_stop and check_stop():
            break
            
        if progress:
            progress(processed_count/total_frames, f"Analyzing frames: {processed_count}/{total_frames}")
        
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
    
    return results

def select_frames_stage1(video_path, output_dir=None, preset="TikTok/Instagram", 
                      sample_rate=None, num_frames=None, min_frame_distance=30, 
                      use_scene_detection=False, progress=None, check_stop=None,
                      batch_size=8, motion_threshold=25):
    """Fast extraction of potential frames from video"""
    temp_video = cv2.VideoCapture(video_path)
    success, first_frame = temp_video.read()
    temp_video.release()
    
    if success:
        verify_face_detection_gpu_usage(first_frame)
    
    if check_stop is None:
        check_stop = lambda: False
    
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
        check_stop=check_stop
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
        check_stop=check_stop
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
    
    if len(frames) == 0 or len(passed_frames) == 0:
        print("No frames were extracted or passed thresholds.")
        return [], fps, 0, None, None

    diverse_frames = []
    selected_indices = []
    
    if use_scene_detection:
        print("Notice: Scene detection is disabled in Stage 1 for performance")
    
    for frame in passed_frames:
        try:
            frame_idx = frames.index(frame["path"])
            
            if all(abs(frame_idx - selected) >= min_frame_distance for selected in selected_indices):
                diverse_frames.append(frame)
                selected_indices.append(frame_idx)
                
            if len(diverse_frames) >= num_frames:
                break
        except ValueError as e:
            # Path not found in frames list
            print(f"Warning: Frame path not found in frames list: {frame['path']}")
            continue
    
    if len(diverse_frames) < num_frames:
        remaining_frames = [f for f in passed_frames if f not in diverse_frames]
        remaining_frames.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        for frame in remaining_frames:
            if frame not in diverse_frames:
                diverse_frames.append(frame)
                
            if len(diverse_frames) >= num_frames:
                break
    
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