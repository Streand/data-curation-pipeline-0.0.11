import os
import gradio as gr
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines"))
from pipelines.videostagensfw import process_nsfw_batch

def UI_video_stage_nsfw(video_dir=None):
    with gr.Tab("NSFW Video Analysis"):
        gr.Markdown("## NSFW Video Content Detection")
        gr.Markdown("### Extract frames containing nudity, revealing clothing, suggestive poses, and NSFW content")

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
        
        # Enhanced controls
        with gr.Row():
            with gr.Column(scale=1):
                nsfw_threshold = gr.Slider(
                    minimum=0.1, maximum=0.8, value=0.4, step=0.05,
                    label="NSFW Detection Threshold",
                    info="Lower = more strict (detect more content as NSFW). Recommended: 0.3-0.5"
                )
            
            with gr.Column(scale=1):
                frame_sample_rate = gr.Slider(
                    minimum=1, maximum=60, value=1, step=1,
                    label="Frame Sample Rate",
                    info="Analyze every Nth frame (1 = every frame, higher = faster but less thorough)"
                )
        
        # NEW: Add checkbox for bounding boxes
        with gr.Row():
            draw_boxes = gr.Checkbox(
                label="Draw Bounding Boxes",
                value=True,
                info="Draw colored boxes around detected NSFW areas (Red=High, Orange=Medium, Yellow=Low)"
            )
        
        # Content type info
        with gr.Accordion("What This Detects", open=False):
            gr.Markdown("""
            **High Priority NSFW (0.9+ score):** Red boxes
            - Exposed breasts, buttocks, genitals
            - Full nudity
            
            **Medium Priority NSFW (0.6-0.9 score):** Orange boxes
            - Covered breasts/buttocks in tight clothing
            - Cleavage, underwear visible
            - Suggestive poses
            
            **Low Priority NSFW (threshold-0.6 score):** Yellow boxes
            - Revealing clothing (crop tops, short skirts)
            - Swimwear, lingerie
            - Exposed midriff
            """)
        
        # Process button
        with gr.Row():
            process_btn = gr.Button("Analyze Videos for NSFW Content", variant="primary")
        
        # Enhanced results display with multiple categories
        with gr.Tabs():
            with gr.TabItem("High NSFW"):
                high_nsfw_gallery = gr.Gallery(
                    label="High NSFW Content (Explicit)", 
                    show_label=True,
                    columns=4,
                    height=300
                )
            
            with gr.TabItem("Medium NSFW"):
                medium_nsfw_gallery = gr.Gallery(
                    label="Medium NSFW Content (Suggestive)", 
                    show_label=True,
                    columns=5,
                    height=300
                )
            
            with gr.TabItem("Low NSFW"):
                low_nsfw_gallery = gr.Gallery(
                    label="Low NSFW Content (Revealing)", 
                    show_label=True,
                    columns=6,
                    height=300
                )
            
            with gr.TabItem("Safe Content"):
                safe_gallery = gr.Gallery(
                    label="Safe Content", 
                    show_label=True,
                    columns=6,
                    height=300
                )
        
        # Detailed results
        result_json = gr.JSON(label="NSFW Analysis Results")
        
        # Process function
        def process_videos_for_nsfw(threshold, sample_rate, draw_bboxes):
            try:
                # Call the function from videostagensfw.py
                results = process_nsfw_batch(
                    video_dir,
                    output_dir=output_dir,
                    nsfw_threshold=threshold,
                    frame_interval=sample_rate,
                    draw_boxes=draw_bboxes  # Pass the checkbox value
                )
                
                # Extract frames for gallery display, separated by classification
                high_nsfw_items = []
                medium_nsfw_items = []
                low_nsfw_items = []
                safe_items = []
                
                if "results" in results:
                    for video_result in results["results"]:
                        if "frames" in video_result:
                            for frame in video_result["frames"][:100]:  # Limit display
                                if "path" in frame and "nsfw_score" in frame:
                                    score = frame["nsfw_score"]
                                    classification = frame.get("classification", "unknown")
                                    categories = frame.get("categories", [])
                                    
                                    # Create detailed label with detected categories
                                    label = f"Score: {score:.3f}"
                                    if categories:
                                        high_priority_cats = [cat["category"] for cat in categories if cat.get("priority") == "high"]
                                        if high_priority_cats:
                                            label += f" | {', '.join(high_priority_cats[:2])}"
                                    
                                    # Sort into appropriate galleries
                                    if classification == "high_nsfw":
                                        high_nsfw_items.append((frame["path"], label))
                                    elif classification == "medium_nsfw":
                                        medium_nsfw_items.append((frame["path"], label))
                                    elif classification == "low_nsfw":
                                        low_nsfw_items.append((frame["path"], label))
                                    else:
                                        safe_items.append((frame["path"], label))
                
                # Create summary message
                total_videos = results.get('total_videos', 0)
                total_high = len(high_nsfw_items)
                total_medium = len(medium_nsfw_items)
                total_low = len(low_nsfw_items)
                total_safe = len(safe_items)
                
                status_msg = f"Analyzed {total_videos} videos. Found: {total_high} high NSFW, {total_medium} medium NSFW, {total_low} low NSFW, {total_safe} safe frames"
                
                return high_nsfw_items, medium_nsfw_items, low_nsfw_items, safe_items, results, status_msg
                
            except Exception as e:
                error_msg = f"Error analyzing videos: {str(e)}"
                return [], [], [], [], {"error": str(e)}, error_msg
        
        # Connect event handler - ADD draw_boxes to inputs
        process_btn.click(
            fn=process_videos_for_nsfw,
            inputs=[nsfw_threshold, frame_sample_rate, draw_boxes],  # Added draw_boxes
            outputs=[high_nsfw_gallery, medium_nsfw_gallery, low_nsfw_gallery, safe_gallery, result_json, status]
        )
        
        return status