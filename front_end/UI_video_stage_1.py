import os
import gradio as gr
import sys
import time
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines"))
from pipelines.videostage1 import VideoProcessor, process_batch

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
    with gr.Tab("Video Stage 1"):
        gr.Markdown("## Video Stage 1 - Frame Extraction with OpenCV & InsightFace")

        # Get GPU info for UI display
        gpu_info = get_gpu_info()
        
        # Display GPU information
        gpu_status = f"**GPU:** {gpu_info['name']} | **CUDA:** {gpu_info['cuda_version']} | **VRAM:** {gpu_info['vram_gb']} GB"
        if not gpu_info["has_gpu"]:
            gpu_status += " | **WARNING:** Processing will be slow without GPU"
        
        gr.Markdown(gpu_status)

        # Debug info - print the video directory path
        print(f"Video directory path: {video_dir}")
        if not video_dir or not os.path.exists(video_dir):
            print(f"WARNING: Video directory doesn't exist: {video_dir}")

        # Default directory if none provided
        if video_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_dir = os.path.join(base_dir, "uploads", "videos")
            print(f"Using default video directory: {video_dir}")
        
        # Set up output directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "store_images", "video_stage_1")
        os.makedirs(output_dir, exist_ok=True)
        
        # Now you can use video_dir to access videos
        video_files = []
        if os.path.exists(video_dir):
            video_files = [f for f in os.listdir(video_dir) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            print(f"Found {len(video_files)} video files in {video_dir}")
        
        # Display info about videos
        if video_files:
            gr.Markdown(f"Found {len(video_files)} videos in directory")
            
            with gr.Row():
                # Left column for controls
                with gr.Column(scale=1):
                    with gr.Group():
                        video_dropdown = gr.Dropdown(
                            choices=video_files,
                            label="Select a video to process",
                            interactive=True
                        )
                        
                        # Presets for quick configuration
                        preset = gr.Radio(
                            ["Default", "High Quality", "Fast Processing"],
                            label="Configuration Preset",
                            value="Default",
                            interactive=True
                        )
                    
                    with gr.Accordion("Advanced Parameters", open=False):
                        frame_interval = gr.Slider(
                            minimum=1, maximum=120, value=30, step=1,
                            label="Frame Interval (process every Nth frame)"
                        )
                        
                        max_frames = gr.Slider(
                            minimum=1, maximum=50, value=10, step=1,
                            label="Maximum Frames to Extract"
                        )
                        
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
                    
                    # Processing status
                    status = gr.Markdown("Select a video and click 'Process' to begin")
                    
                    # Process buttons
                    with gr.Row():
                        process_btn = gr.Button("Process Video", variant="primary")
                        process_all_btn = gr.Button("Process All Videos")
                        open_folder_btn = gr.Button("Open Output Folder")
                    
                # Right column for results
                with gr.Column(scale=2):
                    # Gallery to display images
                    gallery = gr.Gallery(
                        label="Extracted Frames", 
                        show_label=True,
                        columns=3,
                        height=500,
                        object_fit="contain"
                    )
                    
                    with gr.Accordion("Frame Distribution", open=False):
                        plot = gr.Plot(label="Frame Distribution and Quality")
                    
                    with gr.Accordion("Processing Details", open=False):
                        result_json = gr.JSON(label="Processing Result")
            
            # Function to apply presets
            def apply_preset(preset_name):
                if preset_name == "High Quality":
                    return {
                        frame_interval: gr.update(value=10),
                        max_frames: gr.update(value=20),
                        min_distance: gr.update(value=20),
                        face_confidence: gr.update(value=0.95),
                        sharpness_threshold: gr.update(value=120)
                    }
                elif preset_name == "Fast Processing":
                    return {
                        frame_interval: gr.update(value=60),
                        max_frames: gr.update(value=5),
                        min_distance: gr.update(value=45),
                        face_confidence: gr.update(value=0.85),
                        sharpness_threshold: gr.update(value=90)
                    }
                else:  # Default
                    return {
                        frame_interval: gr.update(value=30),
                        max_frames: gr.update(value=10),
                        min_distance: gr.update(value=30),
                        face_confidence: gr.update(value=0.9),
                        sharpness_threshold: gr.update(value=100)
                    }
            
            # Process single video function
            def process_single_video(video_name, interval, max_frames, min_distance, 
                                    face_conf, sharpness_thresh):
                if not video_name:
                    return (
                        [], 
                        {"error": "No video selected"}, 
                        "Error: No video selected", 
                        None
                    )
                
                try:
                    start_time = time.time()
                    
                    # Override constants with UI values
                    # (This assumes your VideoProcessor class accepts these parameters)
                    
                    video_path = os.path.join(video_dir, video_name)
                    processor = VideoProcessor(output_dir=output_dir)
                    
                    # Process the video
                    result = processor.process_video(
                        video_path, 
                        frame_interval=interval,
                        max_frames=max_frames,
                        min_distance=min_distance,
                        face_confidence_threshold=face_conf,
                        sharpness_threshold=sharpness_thresh
                    )
                    
                    # Create gallery items
                    gallery_items = create_gallery_from_results(result)
                    
                    # Create distribution plot
                    plot = create_frame_plot(result)
                    
                    elapsed = time.time() - start_time
                    status_msg = f"Processed video in {elapsed:.2f} seconds. "
                    status_msg += f"Extracted {len(gallery_items)} frames from {result.get('frames_processed', 0)} processed frames."
                    
                    return gallery_items, result, status_msg, plot
                    
                except Exception as e:
                    error_msg = f"Error processing video: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    return [], {"error": error_msg}, error_msg, None
            
            # Process all videos function
            def process_all_videos(interval, max_frames, min_distance, 
                                  face_conf, sharpness_thresh):
                try:
                    start_time = time.time()
                    
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
                    
                    # No gallery for batch processing
                    elapsed = time.time() - start_time
                    status_msg = f"Processed {result.get('total_videos', 0)} videos in {elapsed:.2f} seconds. "
                    status_msg += f"Results saved to {output_dir}"
                    
                    return [], result, status_msg, None
                    
                except Exception as e:
                    error_msg = f"Error processing videos: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    return [], {"error": error_msg}, error_msg, None
            
            # Open folder function
            def open_output_folder():
                return open_folder(output_dir)
            
            # Connect events to UI
            preset.change(
                fn=apply_preset,
                inputs=[preset],
                outputs=[frame_interval, max_frames, min_distance, face_confidence, sharpness_threshold]
            )
            
            # Connect the buttons to the processing functions
            process_btn.click(
                fn=process_single_video,
                inputs=[
                    video_dropdown, frame_interval, max_frames, min_distance,
                    face_confidence, sharpness_threshold
                ],
                outputs=[gallery, result_json, status, plot]
            )
            
            process_all_btn.click(
                fn=process_all_videos,
                inputs=[
                    frame_interval, max_frames, min_distance,
                    face_confidence, sharpness_threshold
                ],
                outputs=[gallery, result_json, status, plot]
            )
            
            open_folder_btn.click(
                fn=open_output_folder,
                inputs=[],
                outputs=[status]
            )
            
        else:
            # If no videos found, show proper message and instructions
            gr.Markdown("""
            ### No videos found in the uploads directory
            
            Please add video files to the uploads/videos directory before processing.
            
            1. Use the 'Upload' tab to upload video files
            2. Or manually copy video files to the 'uploads/videos' directory
            
            Supported formats: .mp4, .avi, .mov, .mkv
            """)
            
            # Add a button to open the uploads folder
            with gr.Row():
                open_uploads_btn = gr.Button("Open Uploads Folder")
                refresh_btn = gr.Button("Refresh")
                
            def refresh_page():
                """Refresh the page to check for new videos"""
                # Re-check for videos
                if os.path.exists(video_dir):
                    video_count = len([f for f in os.listdir(video_dir) 
                                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
                    return f"Refreshed. Found {video_count} videos." 
                return "Refreshed. No videos found."
                
            open_uploads_btn.click(
                fn=lambda: open_folder(video_dir),
                inputs=[],
                outputs=[gr.Markdown(value="")]
            )
            
            refresh_btn.click(
                fn=refresh_page,
                inputs=[],
                outputs=[gr.Markdown(value="")]
            )
