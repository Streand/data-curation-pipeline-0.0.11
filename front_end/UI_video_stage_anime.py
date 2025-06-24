import os
import gradio as gr
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines"))
from pipelines.videostageanime import process_anime_batch_stage1, filter_frames_by_anime_character

def UI_video_stage_anime(video_dir=None):
    with gr.Tab("Anime Processing"):
        gr.Markdown("## Anime Video Frame Extraction")
        gr.Markdown("### Extract high-quality frames from anime videos")

        # Default directory if none provided
        if video_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_dir = os.path.join(base_dir, "uploads", "videos")
        
        # Set up output directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "store_images", "video_stage_anime")
        os.makedirs(output_dir, exist_ok=True)
        
        # Status display
        status = gr.Markdown(f"Ready to process anime videos")
        
        # Basic controls
        with gr.Row():
            frame_interval = gr.Slider(
                minimum=1, maximum=30, value=1, step=1,
                label="Frame Interval (process every Nth frame)"
            )
        
        # Process button
        with gr.Row():
            process_btn = gr.Button("Process Anime Videos", variant="primary")
        
        # Gallery to display extracted frames
        gallery = gr.Gallery(
            label="Extracted Anime Frames", 
            show_label=True,
            columns=6,
            height=400,
            object_fit="cover"
        )
        
        # Results display
        result_json = gr.JSON(label="Processing Results")
        
        # Process function - SIMPLIFIED: Only frame extraction, no character filtering
        def process_anime_videos(interval):
            try:
                # Extract all frames
                stage1_results = process_anime_batch_stage1(
                    video_dir=video_dir,
                    output_dir=output_dir,
                    frame_interval=interval
                )
                
                if "error" in stage1_results:
                    return [], stage1_results, f"Error in frame extraction: {stage1_results['error']}"
                
                # Show regular extracted frames
                gallery_items = []
                for video_result in stage1_results.get("results", []):
                    if video_result.get("status") == "success":
                        for frame in video_result.get("frames", [])[:20]:  # Limit display
                            if "path" in frame:
                                timestamp = frame.get("timestamp", 0)
                                gallery_items.append((frame["path"], f"Time: {timestamp:.2f}s"))
                
                status_msg = f"Processed {stage1_results.get('total_videos', 0)} anime videos. Extracted {stage1_results.get('total_frames_extracted', 0)} frames."
                
                return gallery_items, stage1_results, status_msg
                
            except Exception as e:
                return [], {"error": str(e)}, f"Error processing anime videos: {str(e)}"
        
        # Connect event handler - UPDATED: Only frame_interval input
        process_btn.click(
            fn=process_anime_videos,
            inputs=[frame_interval],
            outputs=[gallery, result_json, status]
        )
        
        return status