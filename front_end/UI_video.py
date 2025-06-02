import gradio as gr
import os
import cv2
import shutil
from pipelines.video import VIDEO_PRESETS, select_best_frames
import matplotlib.pyplot as plt

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
            content_preset = gr.Radio(
                choices=["TikTok/Instagram", "YouTube", "Custom"],
                label="Content Type Preset",
                value="Custom"
            )
        
        with gr.Row():
            sample_rate = gr.Slider(
                minimum=1, maximum=60, value=15, step=1, 
                label="Sample Rate (frames to skip)"
            )
            num_frames = gr.Slider(
                minimum=0, maximum=100, value=20, step=5,
                label="Number of Best Frames (0 = auto based on length)"
            )
        
        with gr.Accordion("Advanced Options", open=False):
            use_clip = gr.Checkbox(label="Use CLIP for aesthetic scoring", value=False)  # Default to unchecked
            
            # Make clip_prompt visible only when CLIP is enabled
            clip_prompt = gr.Textbox(
                label="CLIP Prompt", 
                value="a high quality portrait photo, professional headshot",
                visible=False  # Initially hidden
            )
            
            # Add a visibility dependency
            use_clip.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[use_clip],
                outputs=[clip_prompt]
            )
            
            use_scene_detection = gr.Checkbox(label="Boost frames at scene changes", value=True)
            min_frame_distance = gr.Slider(
                minimum=5, maximum=60, value=30, step=5,
                label="Minimum Frame Distance (for diversity)"
            )
            
        with gr.Accordion("Scoring Parameters", open=False):
            gr.Markdown("### Adjust weights for frame scoring")
            face_weight = gr.Slider(minimum=0.1, maximum=1.0, value=0.4, step=0.05, label="Face Detection Weight")
            sharpness_weight = gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.05, label="Sharpness Weight")
            aesthetic_weight = gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.05, label="Aesthetic (CLIP) Weight")
            size_weight = gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.05, label="Face Size Weight")
            
            gr.Markdown("### Minimum thresholds")
            face_threshold = gr.Slider(minimum=0.7, maximum=0.99, value=0.95, step=0.01, label="Face Confidence Threshold")
        
        # Status display
        status = gr.Markdown("Click 'Process Videos' to extract the best frames")
        
        # Run button
        run_btn = gr.Button("Process Videos")
        
        # Results
        gallery = gr.Gallery(label="Selected Best Frames", show_label=True, columns=4, height=600)
        results_json = gr.JSON(label="Processing Results")
        
        # Add visualization for scores
        with gr.Accordion("Frame Score Analysis", open=False):
            score_chart = gr.Plot(label="Frame Quality Distribution")
        
        # Update UI based on preset selection
        def update_settings_for_preset(preset):
            if preset == "Custom":
                return (
                    gr.update(value=15), gr.update(value=20), gr.update(value=True),
                    gr.update(value="a high quality portrait photo, professional headshot"),
                    gr.update(value=True), gr.update(value=30),
                    gr.update(value=0.4), gr.update(value=0.2), gr.update(value=0.2), 
                    gr.update(value=0.2), gr.update(value=0.95)
                )
            
            preset_config = VIDEO_PRESETS[preset]
            weights = preset_config["scoring_weights"]
            thresholds = preset_config["thresholds"]
            
            return (
                gr.update(value=preset_config["sample_rate"]), 
                gr.update(value=preset_config["number_of_best_frames"]),
                gr.update(value=preset_config["use_clip_aesthetic"]), 
                gr.update(value=preset_config["clip_prompt"]),
                gr.update(value=preset_config["filter_similar_frames"]), 
                gr.update(value=thresholds["minimum_frame_distance"]),
                gr.update(value=weights["face_confidence"]), 
                gr.update(value=weights["sharpness"]), 
                gr.update(value=weights["aesthetic"]), 
                gr.update(value=weights["face_size"]), 
                gr.update(value=thresholds["face_confidence"])
            )
        
        # Create score visualization
        def create_score_plot(results_data):
            if not results_data or "results" not in results_data:
                return None
                
            fig = plt.figure(figsize=(10, 6))
            scores = []
            labels = []
            
            for i, video_result in enumerate(results_data["results"]):
                for frame in video_result.get("best_frames", []):
                    scores.append(frame.get("score", 0))
                    labels.append(frame.get("filename", f"Frame {len(scores)}"))
            
            plt.bar(range(len(scores)), scores)
            plt.xticks(range(len(scores)), labels, rotation=90)
            plt.xlabel("Frame")
            plt.ylabel("Quality Score")
            plt.title("Frame Quality Distribution")
            plt.tight_layout()
            return fig
        
        # Connect preset selection to UI update
        content_preset.change(
            fn=update_settings_for_preset,
            inputs=[content_preset],
            outputs=[
                sample_rate, num_frames, use_clip, clip_prompt, use_scene_detection, 
                min_frame_distance, face_weight, sharpness_weight, aesthetic_weight,
                size_weight, face_threshold
            ]
        )
        
        # Update process_video to include score visualization
        def process_video_with_presets(uploads_dir, preset, sample_rate, num_frames, use_clip, clip_prompt, 
                                      use_scene_detection, min_frame_distance, face_weight, sharpness_weight,
                                      aesthetic_weight, size_weight, face_threshold):
            # Create custom weights and thresholds
            weights = {
                "face_confidence": face_weight,
                "sharpness": sharpness_weight,
                "aesthetic": aesthetic_weight,
                "face_size": size_weight
            }
            
            thresholds = {
                "face_confidence": face_threshold,
                "minimum_frame_distance": min_frame_distance
            }
            
            # Override preset with custom values
            if preset == "Custom":
                VIDEO_PRESETS["Custom"]["sample_rate"] = sample_rate
                VIDEO_PRESETS["Custom"]["number_of_best_frames"] = num_frames
                VIDEO_PRESETS["Custom"]["use_clip_aesthetic"] = use_clip
                VIDEO_PRESETS["Custom"]["clip_prompt"] = clip_prompt
                VIDEO_PRESETS["Custom"]["filter_similar_frames"] = use_scene_detection
                VIDEO_PRESETS["Custom"]["scoring_weights"] = weights
                VIDEO_PRESETS["Custom"]["thresholds"] = thresholds
            
            # Call your existing process_video function with these parameters
            # This already calls select_best_frames internally with the right parameters
            gallery_images, results_json_data, status_text = process_video(
                uploads_dir, 
                sample_rate, 
                num_frames, 
                use_clip, 
                clip_prompt, 
                use_scene_detection
            )
            
            
            # Create visualization
            fig = create_score_plot(results_json_data)
            
            return gallery_images, results_json_data, status_text, fig
        
        # Run video processing when button is clicked
        run_btn.click(
            fn=process_video_with_presets,
            inputs=[
                gr.State(uploads_dir), content_preset, sample_rate, num_frames, 
                use_clip, clip_prompt, use_scene_detection, min_frame_distance,
                face_weight, sharpness_weight, aesthetic_weight, size_weight, face_threshold
            ],
            outputs=[gallery, results_json, status, score_chart]
        )