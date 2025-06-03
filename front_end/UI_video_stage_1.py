import os
import gradio as gr
import sys
import time
import torch
import matplotlib.pyplot as plt
import subprocess

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines"))
from pipelines.videostage1 import process_batch  # Removed unused VideoProcessor import

def get_gpu_info():
    """Get GPU information for display in the UI"""
    info = {
        "has_gpu": False,
        "name": "No GPU detected",
        "cuda_version": "N/A",
        "vram_gb": 0
    }
    
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device)
            memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            
            info["has_gpu"] = True
            info["name"] = device_name
            info["cuda_version"] = torch.version.cuda
            info["vram_gb"] = round(memory_total, 1)
                
        except Exception as e:
            info["error"] = str(e)
    
    return info

def open_folder(folder_path):
    """Open the specified folder in Windows File Explorer"""
    if not folder_path:
        return "No folder specified"
    
    if not os.path.exists(folder_path):
        return f"Folder not found: {folder_path}"
    
    try:
        subprocess.run(['explorer', folder_path])
        return f"Opened folder: {folder_path}"
    except Exception as e:
        return f"Error opening folder: {str(e)}"

def create_frame_plot(results_data):
    """Create a plot showing frame distribution and quality scores"""
    if not results_data or "best_frames" not in results_data:
        return None
        
    fig = plt.figure(figsize=(10, 6))
    
    positions = []
    scores = []
    best_frames = results_data.get("best_frames", [])
    
    if not best_frames:
        return None
    
    for frame in best_frames:
        positions.append(frame.get("frame_num", 0))
        scores.append(frame.get("score", 0))
    
    # Sort based on frame number
    sorted_data = sorted(zip(positions, scores))
    positions = [d[0] for d in sorted_data]
    scores = [d[1] for d in sorted_data]
    
    # Plot frame positions
    plt.subplot(2, 1, 1)
    plt.stem(positions, scores, markerfmt='go', linefmt='g-', basefmt='r-')
    plt.title("Selected Frame Distribution")
    plt.xlabel("Frame Number")
    plt.ylabel("Quality Score")
    
    # Plot scores
    plt.subplot(2, 1, 2)
    plt.bar(range(len(scores)), sorted(scores, reverse=True))
    plt.title("Frame Quality Scores")
    plt.xlabel("Frame Rank")
    plt.ylabel("Quality Score")
    
    plt.tight_layout()
    return fig

def create_gallery_from_results(results):
    """Create gallery items from processing results"""
    if not results or "best_frames" not in results:
        return []
    
    gallery_items = []
    
    for frame in results.get("best_frames", []):
        if "path" in frame:
            try:
                score = round(frame.get("score", 0), 3)
                frame_num = frame.get("frame_num", "?")
                face_count = len(frame.get("faces", []))
                caption = f"Frame: {frame_num} | Score: {score} | Faces: {face_count}"
                gallery_items.append((frame["path"], caption))
            except Exception as e:
                print(f"Error creating gallery item: {e}")
                continue
    
    return gallery_items

def UI_video_stage_1(video_dir=None):
    with gr.Tab("Video Stage 1") as tab:
        gr.Markdown("## Video Stage 1 - Frame Extraction with OpenCV & InsightFace")

        # Get GPU info for UI display
        gpu_info = get_gpu_info()
        
        # Display GPU information
        gpu_status = f"**GPU:** {gpu_info['name']} | **CUDA:** {gpu_info['cuda_version']} | **VRAM:** {gpu_info['vram_gb']} GB"
        if not gpu_info["has_gpu"]:
            gpu_status += " | **WARNING:** Processing will be slow without GPU"
        
        gr.Markdown(gpu_status)

        # Default directory if none provided
        if video_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_dir = os.path.join(base_dir, "uploads", "videos")
        
        # Set up output directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "store_images", "video_stage_1")
        os.makedirs(output_dir, exist_ok=True)
        
        # Now you can use video_dir to access videos
        video_files = []
        if os.path.exists(video_dir):
            video_files = [f for f in os.listdir(video_dir) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        # Create UI state components that will be conditionally shown
        with gr.Row():
            status = gr.Markdown(f"Found {len(video_files)} videos in directory" if video_files else "No videos found")
        
        # Create main UI with buttons at the top
        with gr.Row():
            refresh_btn = gr.Button("Refresh Video List", variant="secondary")
            open_folder_btn = gr.Button("Open Output Folder", variant="secondary") 
            process_btn = gr.Button("Process Videos", variant="primary", size="lg")
        
        # Gallery to display images
        gallery = gr.Gallery(
            label="Extracted Frames", 
            show_label=True,
            columns=3,
            height=500,
            object_fit="contain"
        )
        
        # Processing details and parameters in collapsible accordion
        with gr.Accordion("Processing Details", open=True):
            result_json = gr.JSON(label="Processing Results")
            
            # Add advanced parameters in a nested accordion
            with gr.Accordion("Advanced Parameters", open=False):
                # Presets for quick configuration
                preset = gr.Radio(
                    ["Default", "High Quality", "Fast Processing"],
                    label="Configuration Preset",
                    value="Default",
                    interactive=True
                )
                
                with gr.Row():
                    with gr.Column():
                        frame_interval = gr.Slider(
                            minimum=1, maximum=120, value=30, step=1,
                            label="Frame Interval (process every Nth frame)"
                        )
                        
                        max_frames = gr.Slider(
                            minimum=1, maximum=50, value=10, step=1,
                            label="Maximum Frames to Extract (per video)"
                        )
                    
                    with gr.Column():
                        min_distance = gr.Slider(
                            minimum=5, maximum=120, value=30, step=5,
                            label="Minimum Frame Distance"
                        )
                        
                        face_confidence = gr.Slider(
                            minimum=0.5, maximum=0.99, value=0.9, step=0.01,
                            label="Minimum Face Confidence"
                        )
                        
                        sharpness_threshold = gr.Slider(
                            minimum=50, maximum=200, value=100, step=5,
                            label="Minimum Sharpness"
                        )

        # Function definitions
        def apply_preset(preset_name):
            if preset_name == "High Quality":
                return gr.update(value=10), gr.update(value=20), gr.update(value=20), gr.update(value=0.95), gr.update(value=120)
            elif preset_name == "Fast Processing":
                return gr.update(value=60), gr.update(value=5), gr.update(value=45), gr.update(value=0.85), gr.update(value=90)
            else:  # Default
                return gr.update(value=30), gr.update(value=10), gr.update(value=30), gr.update(value=0.9), gr.update(value=100)
        
        def refresh_videos():
            """Refresh the video list and update count"""
            if os.path.exists(video_dir):
                # Get fresh list of video files
                new_video_files = [f for f in os.listdir(video_dir) 
                              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                
                has_videos = len(new_video_files) > 0
                
                # Print to console for debugging
                print(f"Refresh detected {len(new_video_files)} videos in {video_dir}")
                
                # Force status message to update with current count
                status_msg = f"Found {len(new_video_files)} videos in directory"
                if not has_videos:
                    status_msg = "No videos found"
                
                return status_msg
            return f"Video directory not found: {video_dir}"
        
        def process_videos(interval, max_frames, min_distance, 
                          face_conf, sharpness_thresh):
            try:
                start_time = time.time()
                
                # Get fresh list of videos in case files were added/removed
                if os.path.exists(video_dir):
                    video_count = len([f for f in os.listdir(video_dir) 
                                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
                    if video_count == 0:
                        return [], {"error": "No videos found in directory"}, "No videos found in directory"
                
                # Process all videos
                result = process_batch(
                    video_dir,
                    output_dir=output_dir, 
                    frame_interval=interval,
                    max_frames=max_frames,
                    min_distance=min_distance,
                    face_confidence_threshold=face_conf,
                    sharpness_threshold=sharpness_thresh
                )
                
                # Get a sample of processed frames for gallery display
                gallery_items = []
                if "results" in result and result["results"]:
                    for video_result in result["results"]:
                        if "best_frames" in video_result and video_result["best_frames"]:
                            for frame in video_result["best_frames"][:2]:
                                if "path" in frame:
                                    video_name = os.path.basename(video_result.get("video", "unknown"))
                                    score = round(frame.get("score", 0), 3)
                                    frame_num = frame.get("frame_num", "?")
                                    gallery_items.append((frame["path"], f"{video_name} - Frame {frame_num} (Score: {score})"))
                                    if len(gallery_items) >= 12:
                                        break
                
                elapsed = time.time() - start_time
                status_msg = f"Processed {result.get('total_videos', 0)} videos in {elapsed:.2f} seconds. "
                status_msg += f"Results saved to {output_dir}"
                
                return gallery_items, result, status_msg
                
            except Exception as e:
                error_msg = f"Error processing videos: {str(e)}"
                print(f"ERROR: {error_msg}")
                return [], {"error": error_msg}, error_msg
        
        def open_output_folder():
            return open_folder(output_dir)
        
        # Connect events to UI components
        preset.change(
            fn=apply_preset,
            inputs=[preset],
            outputs=[frame_interval, max_frames, min_distance, face_confidence, sharpness_threshold]
        )
        
        refresh_btn.click(
            fn=refresh_videos,
            inputs=[],
            outputs=[status]
        )
        
        open_folder_btn.click(
            fn=open_output_folder,
            inputs=[],
            outputs=[status]
        )
        
        process_btn.click(
            fn=process_videos,
            inputs=[
                frame_interval, max_frames, min_distance,
                face_confidence, sharpness_threshold
            ],
            outputs=[gallery, result_json, status]
        )
        
        # Always connect tab selection to refresh
        tab.select(
            fn=refresh_videos,
            inputs=[],
            outputs=[status]
        )
        
        return tab
