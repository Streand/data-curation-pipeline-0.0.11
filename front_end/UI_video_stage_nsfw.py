import os
import gradio as gr
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines"))
from pipelines.videostagensfw import process_nsfw_analysis

def UI_video_stage_nsfw(video_dir=None):
    with gr.Tab("NSFW Video Analysis"):
        gr.Markdown("## NSFW Video Content Analysis")
        gr.Markdown("### Detect and filter NSFW content in videos")

        # Default directory if none provided
        if video_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_dir = os.path.join(base_dir, "uploads", "videos")
        
        # Set up output directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "store_images", "video_stage_nsfw")
        os.makedirs(output_dir, exist_ok=True)
        
        # Status display
        status = gr.Markdown(f"Ready to analyze videos for NSFW content")
        
        # Controls
        with gr.Row():
            with gr.Column(scale=1):
                nsfw_threshold = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="NSFW Detection Threshold",
                    info="Lower values are more strict (detect more content as NSFW)"
                )
            
            with gr.Column(scale=1):
                frame_sample_rate = gr.Slider(
                    minimum=1, maximum=30, value=10, step=1,
                    label="Frame Sample Rate",
                    info="Analyze every Nth frame (higher = faster but less thorough)"
                )
        
        # Process button
        with gr.Row():
            process_btn = gr.Button("Analyze Videos for NSFW Content", variant="primary")
        
        # Results display with filtered/categorized images
        with gr.Tabs():
            with gr.TabItem("Safe Content"):
                safe_gallery = gr.Gallery(
                    label="Safe Content Frames", 
                    show_label=True,
                    columns=6,
                    height=300
                )
            
            with gr.TabItem("NSFW Content"):
                nsfw_gallery = gr.Gallery(
                    label="NSFW Content Frames", 
                    show_label=True,
                    columns=6,
                    height=300
                )
        
        # Detailed results
        result_json = gr.JSON(label="NSFW Analysis Results")
        
        # Process function (implement in videostagensfw.py)
        def process_videos_for_nsfw(threshold, sample_rate):
            try:
                # Call the function from videostagensfw.py
                results = process_nsfw_analysis(
                    video_dir,
                    output_dir=output_dir,
                    nsfw_threshold=threshold,
                    frame_sample_rate=sample_rate
                )
                
                # Extract frames for gallery display, separated by classification
                safe_items = []
                nsfw_items = []
                
                if "results" in results:
                    for video_result in results["results"]:
                        if "frames" in video_result:
                            for frame in video_result["frames"][:50]:  # Limit display
                                if "path" in frame and "nsfw_score" in frame:
                                    score = frame["nsfw_score"]
                                    label = f"NSFW Score: {score:.3f}"
                                    
                                    if score < threshold:
                                        safe_items.append((frame["path"], label))
                                    else:
                                        nsfw_items.append((frame["path"], label))
                
                return safe_items, nsfw_items, results, f"Analyzed {results.get('total_videos', 0)} videos"
                
            except Exception as e:
                return [], [], {"error": str(e)}, f"Error analyzing videos: {str(e)}"
        
        # Connect event handler
        process_btn.click(
            fn=process_videos_for_nsfw,
            inputs=[nsfw_threshold, frame_sample_rate],
            outputs=[safe_gallery, nsfw_gallery, result_json, status]
        )
        
        return status