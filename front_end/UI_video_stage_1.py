import os
import gradio as gr
import sys
import time
import subprocess

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines"))
from pipelines.videostage1 import process_batch

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

def UI_video_stage_1(video_dir=None):
    with gr.Tab("Video Stage 1") as tab:
        gr.Markdown("## Video Frame Extraction")
        gr.Markdown("Extract all frames with faces from videos")

        # Default directory if none provided
        if video_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_dir = os.path.join(base_dir, "uploads", "videos")
        
        # Set up output directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "store_images", "video_stage_1")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video files
        video_files = []
        if os.path.exists(video_dir):
            video_files = [f for f in os.listdir(video_dir) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        # Status display
        with gr.Row():
            status = gr.Markdown(f"Found {len(video_files)} videos in directory" if video_files else "No videos found")
        
        # Simple controls - just frame interval and a checkbox for all frames
        with gr.Row():
            frame_interval = gr.Slider(
                minimum=1, maximum=30, value=5, step=1,
                label="Frame Interval (process every Nth frame, lower = more frames)"
            )
        
        # Buttons
        with gr.Row():
            refresh_btn = gr.Button("Refresh Video List", variant="secondary")
            open_folder_btn = gr.Button("Open Output Folder", variant="secondary") 
            process_btn = gr.Button("Extract All Frames", variant="primary", size="lg")
        
        # Gallery to display extracted frames with smaller thumbnails
        gallery = gr.Gallery(
            label="Extracted Frames", 
            show_label=True,
            columns=6,  # Increase from 3 to 6 images per row
            height=400,  # Slightly reduce height to fit UI better
            object_fit="cover",  # Change from "contain" to "cover" for better thumbnails
            preview=True,  # Ensure clicking opens a larger view
            allow_preview=True,  # Explicitly allow preview
            elem_id="frame_gallery"  # Add ID for potential custom styling
        )
        
        # Result information
        with gr.Accordion("Processing Results", open=True):
            result_json = gr.JSON(label="Processing Summary")

        # Function to refresh video list
        def refresh_videos():
            if os.path.exists(video_dir):
                new_video_files = [f for f in os.listdir(video_dir) 
                              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                
                print(f"Refresh detected {len(new_video_files)} videos in {video_dir}")
                
                status_msg = f"Found {len(new_video_files)} videos in directory"
                if not len(new_video_files):
                    status_msg = "No videos found"
                
                return status_msg
            return f"Video directory not found: {video_dir}"
        
        # Function to process videos - simplified to always extract all frames
        def process_videos(interval):
            try:
                start_time = time.time()
                
                if not os.path.exists(video_dir):
                    return [], {"error": f"Video directory not found: {video_dir}"}, f"Video directory not found: {video_dir}"
                
                video_count = len([f for f in os.listdir(video_dir) 
                                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
                if video_count == 0:
                    return [], {"error": "No videos found in directory"}, "No videos found in directory"
                
                # Process all videos - use simplified parameters
                result = process_batch(
                    video_dir,
                    output_dir=output_dir, 
                    frame_interval=interval,
                    max_frames=9999,               # Set very high to get all frames
                    min_distance=1,                # No minimum distance between frames
                    face_confidence_threshold=0.5, # Low threshold to accept most faces
                    sharpness_threshold=10,        # Low threshold for sharpness
                    extract_all_good_frames=True   # Always extract all frames that pass thresholds
                )
                
                # Get ALL processed frames for gallery (no limits)
                gallery_items = []
                if "results" in result and result["results"]:
                    for video_result in result["results"]:
                        if "best_frames" in video_result and video_result["best_frames"]:
                            video_name = os.path.basename(video_result.get("video", "unknown"))
                            # Show ALL frames from the video (removed the [:4] limit)
                            for frame in video_result["best_frames"]:
                                if "path" in frame:
                                    frame_num = frame.get("frame_num", "?")
                                    gallery_items.append((frame["path"], f"{video_name} - Frame {frame_num}"))
                            # Removed the break condition that limited to 12 frames total

                elapsed = time.time() - start_time
                total_frames = sum(len(r.get("best_frames", [])) for r in result.get("results", []))
                status_msg = f"Processed {result.get('total_videos', 0)} videos in {elapsed:.2f} seconds. "
                status_msg += f"Extracted {total_frames} frames. Saved to {output_dir}"
                
                # Warning if there are many frames
                if len(gallery_items) > 50:
                    status_msg += f"\nShowing all {len(gallery_items)} frames in preview (may be slow)"
                
                return gallery_items, result, status_msg
                
            except Exception as e:
                error_msg = f"Error processing videos: {str(e)}"
                print(f"ERROR: {error_msg}")
                return [], {"error": error_msg}, error_msg
        
        # Function to open output folder
        def open_output_folder():
            return open_folder(output_dir)
        
        # Connect events to UI components
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
            inputs=[frame_interval],
            outputs=[gallery, result_json, status]
        )
        
        # Refresh on tab selection
        tab.select(
            fn=refresh_videos,
            inputs=[],
            outputs=[status]
        )
        
        return tab
