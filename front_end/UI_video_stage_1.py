import gradio as gr
import os
import cv2
import shutil
from pipelines.videostage1 import VIDEO_PRESETS, select_frames_stage1 as select_best_frames
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import time

# 1. Add a shared state variable at the top of the file (after imports)
# This will keep track of whether processing should be interrupted
STOP_PROCESSING = False

# 2. Add function to handle the stop button click
def stop_video_processing():
    """Stop the current video processing"""
    global STOP_PROCESSING
    STOP_PROCESSING = True
    return "Processing will stop after current frame. Please wait..."

# 3. Modify the process_video_stage1 function to check for stop flag
def process_video_stage1(uploads_dir, preset, sample_rate, num_frames, min_frame_distance, use_scene_detection, progress=gr.Progress()):
    global STOP_PROCESSING
    STOP_PROCESSING = False  # Reset flag at start
    
    from pipelines.videostage1 import select_frames_stage1
    
    # Start the timer
    start_time = time.time()
    
    status_text = "Processing videos..."
    results_json_data = {"results": []}
    gallery_images = []
    
    # Initialize output paths to avoid UnboundLocalError
    storage_path = None
    best_frames_path = None
    
    # Check if directory exists and contains videos
    if not os.path.exists(uploads_dir):
        return [], {"error": "Upload directory not found"}, "Error: Upload directory not found"
    
    # Process each video in uploads directory
    video_files = [f for f in os.listdir(uploads_dir) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
    
    if not video_files:
        return [], {"error": "No video files found"}, "Error: No video files found in upload directory"
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(os.path.dirname(uploads_dir), "frame_output")
    
    for video_file in video_files:
        try:
            video_path = os.path.join(uploads_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_dir = os.path.join(output_base, f"{video_name}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Update status
            progress(0, desc=f"Processing {video_file}...")
            
            # Run stage 1 processing with progress reporting
            result_frames, fps, total_frames, storage_path, best_frames_path = select_frames_stage1(
                video_path, 
                preset=preset,
                sample_rate=sample_rate, 
                num_frames=num_frames, 
                min_frame_distance=min_frame_distance,
                use_scene_detection=use_scene_detection,
                progress=progress,
                check_stop=lambda: STOP_PROCESSING,
                batch_size=8,  # Default batch size
                motion_threshold=25  # Default motion threshold
            )
            
            # Add error handling for None return values
            if not result_frames or best_frames_path is None:
                raise Exception(f"No valid frames could be extracted from {video_file}")
                
            # Update the video result to include storage path
            video_result = {
                "video": video_file,
                "output_dir": storage_path,  # This is now the permanent storage path
                "fps": fps,
                "total_frames": total_frames,
                "best_frames": result_frames
            }
            
            results_json_data["results"].append(video_result)
            
            # Add images to gallery
            for frame in result_frames:
                score = round(frame.get("score", 0), 3)
                faces = frame.get("faces", 0)
                gallery_images.append((frame["path"], f"Score: {score}, Faces: {faces}"))
            
            status_text = f"Successfully processed {len(video_files)} videos, extracted {len(gallery_images)} candidate frames"
            
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            status_text = f"Error processing {video_file}: {str(e)}"
            # Initialize paths to default values if processing failed
            storage_path = os.path.join(get_app_root(), "store_images", "video_stage_1")
            best_frames_path = os.path.join(get_app_root(), "store_images", "video_stage_1_best")
    
    # Create the visualization
    fig = create_frame_plot(results_json_data)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    # Format time as min:sec
    time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds} seconds"
    
    # Add processing time to status
    status_text += f"\nTotal processing time: {time_str}"
    
    # Add storage location to status
    if 'storage_path' in locals() and storage_path:
        status_text += f"\nFrames saved to: {os.path.dirname(os.path.dirname(storage_path))}"
    
    # Add a note if processing was stopped early
    if STOP_PROCESSING:
        status_text = f"Processing stopped early. Showing best {len(gallery_images)} frames found so far."
    

# 1. Fix the process_video function definition (remove CLIP parameters)
def process_video(uploads_dir, sample_rate, num_frames, use_scene_detection=False):
    # Get all video files from uploads
    video_files = [f for f in os.listdir(uploads_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        return None, None, "No video files found in uploads folder."
    
    # Create temp folder for frames
    temp_dir = os.path.join(uploads_dir, "video_frames_temp")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    results = []
    gallery_images = []
    total_frames_processed = 0
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(uploads_dir, video_file)
        video_temp_dir = os.path.join(temp_dir, os.path.splitext(video_file)[0])
        os.makedirs(video_temp_dir, exist_ok=True)
        
        try:
            # Select best frames using GPU acceleration (removed CLIP parameters)
            best_frames, fps, total_frames = select_best_frames(
                video_path, 
                video_temp_dir, 
                sample_rate=sample_rate,
                num_frames=num_frames,
                use_scene_detection=use_scene_detection
            )
            
            total_frames_processed += total_frames
            
            # Add to gallery for display
            for frame in best_frames:
                frame_path = frame["path"]
                img = cv2.imread(frame_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert for display
                    # Add score as text on image
                    score_text = f"Score: {frame['score']:.2f}"
                    cv2.putText(img, score_text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    gallery_images.append((img, f"{os.path.basename(frame_path)} - {score_text}"))
            
            # Add results for this video
            results.append({
                "video": video_file,
                "frames_processed": total_frames,
                "frames_selected": len(best_frames),
                "fps": fps,
                "best_frames": [
                    {
                        "filename": os.path.basename(f["path"]),
                        "score": f["score"],
                        "has_face": f.get("has_face", False)
                    } 
                    for f in best_frames
                ]
            })
            
        except Exception as e:
            results.append({
                "video": video_file,
                "error": str(e)
            })
    
    status_text = f"Processed {len(video_files)} videos with {total_frames_processed} total frames. Selected {len(gallery_images)} best frames."
    return gallery_images, {"summary": status_text, "results": results}, status_text

def open_folder(folder_path):
    """Open the specified folder in the file explorer"""
    if not folder_path:
        return "No folder specified"
    
    if not os.path.exists(folder_path):
        return f"Folder not found: {folder_path}"
    
    try:
        if os.name == 'nt':  # Windows
            # Use subprocess with SW_SHOWMAXIMIZED (3) flag instead of os.startfile
            subprocess.run(['explorer', '/select,', folder_path], shell=True)
        elif os.name == 'posix':  # macOS/Linux
            if os.path.exists('/usr/bin/open'):  # macOS
                subprocess.call(['open', folder_path])
            else:  # Linux
                subprocess.call(['xdg-open', folder_path])
        return f"Opened folder: {folder_path}"
    except Exception as e:
        return f"Error opening folder: {str(e)}"

def get_app_root():
    """Get the application root directory"""
    current_file = os.path.abspath(__file__)
    front_end_dir = os.path.dirname(current_file)
    app_root = os.path.dirname(front_end_dir)
    return app_root

def open_all_frames_folder():
    """Open the all frames folder at the known location"""
    all_frames_dir = os.path.join(get_app_root(), "store_images", "video_stage_1")
    return open_folder(all_frames_dir)

def create_frame_plot(results_data):
    if not results_data or "results" not in results_data:
        return None
        
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig = plt.figure(figsize=(10, 6))
    
    # Extract frame positions and scores
    positions = []
    scores = []
    passed = []
    
    for video_result in results_data.get("results", []):
        for frame in video_result.get("best_frames", []):
            frame_path = frame.get("path", "")
            # Extract frame number from filename (assuming format frame_XXXX.jpg)
            try:
                frame_num = int(os.path.basename(frame_path).split('_')[1].split('.')[0])
                positions.append(frame_num)
                scores.append(frame.get("score", 0))
                passed.append(frame.get("passed_threshold", False))
            except:
                continue
    
    if not positions:
        return None
        
    # Sort by position
    sorted_data = sorted(zip(positions, scores, passed))
    positions = [d[0] for d in sorted_data]
    scores = [d[1] for d in sorted_data]
    passed = [d[2] for d in sorted_data]
    
    # Plot frame distribution
    plt.subplot(2, 1, 1)
    plt.scatter(positions, [1]*len(positions), c=['green' if p else 'red' for p in passed], 
              alpha=0.7, s=50)
    plt.ylabel("Selected")
    plt.title("Frame Distribution")
    
    # Plot scores
    plt.subplot(2, 1, 2)
    plt.bar(range(len(scores)), scores, color=['green' if p else 'red' for p in passed])
    plt.xlabel("Frame Index")
    plt.ylabel("Quality Score")
    plt.title("Frame Scores")
    
    plt.tight_layout()
    return fig

def video_tab_stage1(uploads_dir):
    with gr.Tab("Video Processing - Stage 1"):
        gr.Markdown("""
        ## Stage 1: Fast Frame Extraction
        This is the first stage of processing, designed to quickly extract potentially good frames 
        with permissive thresholds. Only basic OpenCV and InsightFace models are used for speed.
        """)
        
        with gr.Row():
            content_preset = gr.Radio(
                choices=["TikTok/Instagram", "YouTube", "Custom"],
                label="Content Type Preset",
                value="TikTok/Instagram"
            )
        
        with gr.Row():
            sample_rate = gr.Slider(
                minimum=1, maximum=60, value=1, step=1,  # Change default from 15 to 1
                label="Sample Rate (frames to skip)"
            )
            num_frames = gr.Slider(
                minimum=5, maximum=100, value=8, step=5,  # Change default from 20 to 8
                label="Maximum Number of Frames to Extract"
            )
            
        with gr.Row():
            min_frame_distance = gr.Slider(
                minimum=5, maximum=60, value=15, step=5,  # Change default from 30 to 15
                label="Minimum Frame Distance (for diversity)"
            )
            use_scene_detection = gr.Checkbox(
                label="Try to detect scene changes (optional, may be slower)", 
                value=False
            )
        
        # Add this to your video_tab function:
        with gr.Row():
            batch_size = gr.Slider(
                minimum=1, maximum=32, value=8, step=1,
                label="Batch Size (higher uses more GPU memory)"
            )
            motion_threshold = gr.Slider(
                minimum=0, maximum=50, value=25, step=5,
                label="Motion Threshold (higher = fewer similar frames)"
            )
        
        # Storage location display
        store_path = gr.Textbox(
            label="Storage Location", 
            value=os.path.join(os.path.dirname(uploads_dir), "store_images", "video_stage_1"),
            interactive=False
        )
        
        # Status display
        status = gr.Markdown("Click 'Process Videos' to extract potential frames")
        
        # Button row - put all buttons in the same row
        with gr.Row():
            all_frames_btn = gr.Button("Open All Frames Folder", variant="secondary", scale=1)
            open_btn = gr.Button("Open Best Frames Folder", variant="secondary", scale=1)
            stop_btn = gr.Button("Stop", variant="stop", scale=1)  # Add the stop button
            run_btn = gr.Button("Process Videos", variant="primary", scale=2)
        
        # Results
        gallery = gr.Gallery(label="Selected Candidate Frames", show_label=True, columns=4, height=600)
        results_json = gr.JSON(label="Processing Results")
        
        # Add visualization for frames
        with gr.Accordion("Frame Distribution", open=False):
            dist_chart = gr.Plot(label="Frame Distribution")
        
        # Update settings based on preset selection
        def update_settings_for_preset(preset):
            if preset == "TikTok/Instagram":
                return (
                    gr.update(value=1), 
                    gr.update(value=8),
                    gr.update(value=15)
                )
            elif preset == "YouTube":
                return (
                    gr.update(value=5), 
                    gr.update(value=20),
                    gr.update(value=30)
                )
            else:  # Custom
                return (
                    gr.update(value=15), 
                    gr.update(value=20),
                    gr.update(value=30)
                )
        
        # Connect preset selection to UI update
        content_preset.change(
            fn=update_settings_for_preset,
            inputs=[content_preset],
            outputs=[sample_rate, num_frames, min_frame_distance]
        )
        
        # Process video function
        def process_video_stage1(uploads_dir, preset, sample_rate, num_frames, min_frame_distance, use_scene_detection, progress=gr.Progress()):
            global STOP_PROCESSING
            STOP_PROCESSING = False  # Reset flag at start
            
            from pipelines.videostage1 import select_frames_stage1
            
            # Start the timer
            start_time = time.time()
            
            status_text = "Processing videos..."
            results_json_data = {"results": []}
            gallery_images = []
            
            # Initialize output paths to avoid UnboundLocalError
            storage_path = None
            best_frames_path = None
            
            # Check if directory exists and contains videos
            if not os.path.exists(uploads_dir):
                return [], {"error": "Upload directory not found"}, "Error: Upload directory not found"
            
            # Process each video in uploads directory
            video_files = [f for f in os.listdir(uploads_dir) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
            
            if not video_files:
                return [], {"error": "No video files found"}, "Error: No video files found in upload directory"
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = os.path.join(os.path.dirname(uploads_dir), "frame_output")
            
            for video_file in video_files:
                try:
                    video_path = os.path.join(uploads_dir, video_file)
                    video_name = os.path.splitext(video_file)[0]
                    output_dir = os.path.join(output_base, f"{video_name}_{timestamp}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Update status
                    progress(0, desc=f"Processing {video_file}...")
                    
                    # Run stage 1 processing with progress reporting
                    result_frames, fps, total_frames, storage_path, best_frames_path = select_frames_stage1(
                        video_path, 
                        preset=preset,
                        sample_rate=sample_rate, 
                        num_frames=num_frames, 
                        min_frame_distance=min_frame_distance,
                        use_scene_detection=use_scene_detection,
                        progress=progress,
                        check_stop=lambda: STOP_PROCESSING  # Pass a function to check stop status
                    )
                    
                    # Add error handling for None return values
                    if not result_frames or best_frames_path is None:
                        raise Exception(f"No valid frames could be extracted from {video_file}")
                        
                    # Update the video result to include storage path
                    video_result = {
                        "video": video_file,
                        "output_dir": storage_path,  # This is now the permanent storage path
                        "fps": fps,
                        "total_frames": total_frames,
                        "best_frames": result_frames
                    }
                    
                    results_json_data["results"].append(video_result)
                    
                    # Add images to gallery
                    for frame in result_frames:
                        score = round(frame.get("score", 0), 3)
                        faces = frame.get("faces", 0)
                        gallery_images.append((frame["path"], f"Score: {score}, Faces: {faces}"))
                    
                    status_text = f"Successfully processed {len(video_files)} videos, extracted {len(gallery_images)} candidate frames"
                    
                except Exception as e:
                    print(f"Error processing {video_file}: {e}")
                    status_text = f"Error processing {video_file}: {str(e)}"
                    # Initialize paths to default values if processing failed
                    storage_path = os.path.join(get_app_root(), "store_images", "video_stage_1")
                    best_frames_path = os.path.join(get_app_root(), "store_images", "video_stage_1_best")
            
            # Create the visualization
            fig = create_frame_plot(results_json_data)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            
            # Format time as min:sec
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds} seconds"
            
            # Add processing time to status
            status_text += f"\nTotal processing time: {time_str}"
            
            # Add storage location to status
            if 'storage_path' in locals() and storage_path:
                status_text += f"\nFrames saved to: {os.path.dirname(os.path.dirname(storage_path))}"
            
            # Add a note if processing was stopped early
            if STOP_PROCESSING:
                status_text = f"Processing stopped early. Showing best {len(gallery_images)} frames found so far."
            
            # Update the return statement in process_video_stage1:
            return (
                gallery_images, 
                results_json_data, 
                status_text, 
                fig, 
                gr.update(value=best_frames_path if best_frames_path else "")  # Handle None value
            )
        
        # Update the run_btn.click outputs:
        run_btn.click(
            fn=process_video_stage1,
            inputs=[
                gr.State(uploads_dir), content_preset, sample_rate, num_frames,
                min_frame_distance, use_scene_detection
            ],
            outputs=[gallery, results_json, status, dist_chart, store_path]
        )
        
        # Add the click handler for the open folder button
        open_btn.click(
            fn=open_folder,
            inputs=[store_path],
            outputs=[status]
        )
        
        # Add the click handler for the all frames button
        all_frames_btn.click(
            fn=open_all_frames_folder,
            inputs=[],
            outputs=[status]
        )
        
        # Connect the stop button to the stop_video_processing function
        stop_btn.click(
            fn=stop_video_processing,
            inputs=[],
            outputs=[status]
        )