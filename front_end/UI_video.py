import gradio as gr
import os
import cv2
import shutil
from pipelines.video import select_best_frames

def process_video(uploads_dir, sample_rate, num_frames, use_clip=True, clip_prompt="a high quality portrait photo, professional headshot", use_scene_detection=True):
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
            # Select best frames using GPU acceleration
            best_frames, fps, total_frames = select_best_frames(
                video_path, 
                video_temp_dir, 
                sample_rate=sample_rate,
                num_frames=num_frames,
                use_clip=use_clip,
                clip_prompt=clip_prompt,
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

def video_tab(uploads_dir):
    with gr.Tab("Video Tab"):
        gr.Markdown("## Video Frame Selection")
        
        with gr.Row():
            sample_rate = gr.Slider(
                minimum=1, maximum=60, value=15, step=1, 
                label="Sample Rate (frames to skip)"
            )
            num_frames = gr.Slider(
                minimum=5, maximum=100, value=20, step=5,
                label="Number of Best Frames to Select"
            )
        
        with gr.Accordion("Advanced Options", open=False):
            use_clip = gr.Checkbox(label="Use CLIP for aesthetic scoring", value=True)
            clip_prompt = gr.Textbox(
                label="CLIP Prompt", 
                value="a high quality portrait photo, professional headshot", 
                visible=True
            )
            use_scene_detection = gr.Checkbox(label="Filter similar frames", value=True)
        
        # Status display
        status = gr.Markdown("Click 'Process Videos' to extract the best frames from uploaded videos")
        
        # Run button
        run_btn = gr.Button("Process Videos")
        
        # Results
        gallery = gr.Gallery(label="Selected Best Frames", show_label=True, columns=4, height=600)
        results_json = gr.JSON(label="Processing Results")
        
        # Run video processing when button is clicked
        run_btn.click(
            fn=lambda sr, nf, clip, prompt, scene: process_video(
                uploads_dir, sr, nf, use_clip=clip, clip_prompt=prompt, use_scene_detection=scene
            ),
            inputs=[sample_rate, num_frames, use_clip, clip_prompt, use_scene_detection],
            outputs=[gallery, results_json, status]
        )