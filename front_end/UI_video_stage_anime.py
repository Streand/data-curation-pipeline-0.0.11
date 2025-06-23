import os
import gradio as gr
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines"))
from pipelines.videostageanime import process_anime_batch

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
        
        # Controls, buttons, etc. similar to UI_video_stage_1.py
        # Status display
        status = gr.Markdown(f"Ready to process anime videos")
        
        # Basic controls
        with gr.Row():
            frame_interval = gr.Slider(
                minimum=1, maximum=30, value=3, step=1,
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
        
        # Process function (implement in videostageanime.py)
        def process_anime_videos(interval):
            try:
                # Call the function from videostageanime.py
                results = process_anime_batch(
                    video_dir,
                    output_dir=output_dir,
                    frame_interval=interval
                )
                
                # Extract frames for gallery display
                gallery_items = []
                if "results" in results:
                    for video_result in results["results"]:
                        if "frames" in video_result:
                            for frame in video_result["frames"][:20]:  # Limit display
                                if "path" in frame:
                                    gallery_items.append((frame["path"], f"Score: {frame.get('score', 0):.2f}"))
                
                return gallery_items, results, f"Processed {results.get('total_videos', 0)} anime videos"
                
            except Exception as e:
                return [], {"error": str(e)}, f"Error processing anime videos: {str(e)}"
        
        # Connect event handler
        process_btn.click(
            fn=process_anime_videos,
            inputs=[frame_interval],
            outputs=[gallery, result_json, status]
        )
        
        return status