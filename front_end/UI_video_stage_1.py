import os
import gradio as gr
import sys
import time
import subprocess
import cv2
import numpy as np

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
        gr.Markdown("### Extract all frames with faces from videos")
        gr.Markdown("<span style='color: var(--primary-500);'>Preview does not include all frames, only a sample for performance, full frames are saved to the output directory</span>")

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
        
        # Gallery to display extracted frames
        gallery = gr.Gallery(
            label="Extracted Frames", 
            show_label=True,
            columns=6,
            height=400,
            object_fit="cover",
            preview=True,
            allow_preview=True,
            elem_id="frame_gallery"
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
                if not new_video_files:
                    status_msg = "No videos found"
                
                return status_msg
            return f"Video directory not found: {video_dir}"
        
        # Function to process videos - SIMPLIFIED to reliably extract frames
        def process_videos(interval):
            try:
                start_time = time.time()
                
                if not os.path.exists(video_dir):
                    return [], {"error": f"Video directory not found: {video_dir}"}, f"Video directory not found: {video_dir}"
                
                video_count = len([f for f in os.listdir(video_dir) 
                                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
                if video_count == 0:
                    return [], {"error": "No videos found in directory"}, "No videos found in directory"
                
                # Process all videos with more permissive parameters
                result = process_batch(
                    video_dir,
                    output_dir=output_dir, 
                    frame_interval=interval,
                    max_frames=10000,              # Very high to get all frames
                    min_distance=1,                # No minimum distance between frames
                    face_confidence_threshold=0.1, # Very low threshold to accept most faces
                    sharpness_threshold=1,         # Very low threshold for sharpness
                    extract_all_good_frames=True   # Extract all frames that pass the minimal thresholds
                )
                
                # Display frames with face detection overlays
                gallery_items = []

                # Create overlay directory
                overlay_dir = os.path.join(output_dir, "overlay_preview")
                os.makedirs(overlay_dir, exist_ok=True)

                if "results" in result and result["results"]:
                    for video_result in result["results"]:
                        if "best_frames" in video_result and video_result["best_frames"]:
                            video_name = os.path.basename(video_result.get("video", "unknown"))
                            
                            # Take up to the first 10 frames from each video for preview
                            for frame in video_result["best_frames"][:10]:
                                if "path" in frame and "faces" in frame:
                                    frame_path = frame["path"]
                                    frame_num = frame.get("frame_num", "unknown")
                                    
                                    # Create overlay image
                                    overlay_path = os.path.join(overlay_dir, f"overlay_{os.path.basename(frame_path)}")
                                    
                                    # Generate the overlay image
                                    try:
                                        # Create face overlay using the existing function
                                        img = cv2.imread(frame_path)
                                        if img is not None:
                                            # Draw detection overlays
                                            for face in frame["faces"]:
                                                # Access bbox as dictionary
                                                if "bbox" in face:
                                                    bbox = face["bbox"]
                                                    if isinstance(bbox, list) and len(bbox) >= 4:
                                                        x1, y1, x2, y2 = [int(coord) for coord in bbox]
                                                        
                                                        # Draw rectangle
                                                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                                        
                                                        # Display confidence score
                                                        score = round(face.get("score", 0), 3)
                                                        cv2.putText(img, f"{score}", (x1, y1-10),
                                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                            
                                            # Save overlay image
                                            cv2.imwrite(overlay_path, img)
                                            
                                            # Use the overlay in gallery
                                            gallery_items.append((overlay_path, f"{video_name} - Frame {frame_num}"))
                                        else:
                                            # Fallback to original image
                                            gallery_items.append((frame_path, f"{video_name} - Frame {frame_num}"))
                                    except Exception as e:
                                        print(f"Error creating overlay for {frame_path}: {str(e)}")
                                        # Fallback to original image
                                        gallery_items.append((frame_path, f"{video_name} - Frame {frame_num}"))
                                    
                                    # Limit gallery items to prevent UI slowdown
                                    if len(gallery_items) >= 50:
                                        break
                
                elapsed = time.time() - start_time
                total_frames = sum(len(r.get("best_frames", [])) for r in result.get("results", []))
                status_msg = f"Processed {result.get('total_videos', 0)} videos in {elapsed:.2f} seconds. "
                status_msg += f"Extracted {total_frames} frames. Saved to {output_dir}"
                
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

def create_face_overlay(image_path, face_data):
    """
    Create a copy of the image with face detection overlays.
    
    Args:
        image_path: Path to the original image
        face_data: Face detection data from InsightFace
        
    Returns:
        Image with face detection overlays
    """
    # Read the original image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Draw bbox around each face
    for face in face_data:
        # Extract bbox coordinates and convert to integers
        bbox = face.get('bbox')
        if not bbox:
            continue
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Draw rectangle around face
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add confidence score text
        score = face.get('det_score', 0)
        cv2.putText(img, f"{score:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw facial landmarks if available
        landmarks = face.get('landmark_2d_106')
        if landmarks is not None:
            for point in landmarks:
                x, y = int(point[0]), int(point[1])
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    
    # Convert from BGR to RGB for Gradio display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb
