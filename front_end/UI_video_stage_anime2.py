import os
import gradio as gr
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines"))
from pipelines.videostageanime import process_anime_batch_stage1
from pipelines.videostageanime2 import process_character_filtering_batch_gpu as process_character_filtering_batch, get_available_extracted_frames, preview_character_matches

def UI_video_stage_anime2(video_dir=None):
    with gr.Tab("Anime Video Processing"):
        gr.Markdown("## Anime Character Recognition")
        gr.Markdown("### Find specific anime characters in extracted frames")
        
        # Default directories
        if video_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_dir = os.path.join(base_dir, "uploads", "videos")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "store_images", "video_stage_anime")
        os.makedirs(output_dir, exist_ok=True)
        
        # Global status
        global_status = gr.Markdown("Ready to search for anime characters")
        
        # Character Recognition (former Stage 2, now the main feature)
        with gr.Accordion("Character Recognition", open=True):
            gr.Markdown("**Filter extracted frames to find specific anime characters**")
            gr.Markdown("*⚠️ Requires frames to be extracted first using 'Anime Processing (Simple)' tab*")
            
            # Check for available extracted frames
            available_status = gr.Markdown("Click 'Refresh Available Frames' to check for extracted frames")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Upload Reference Character Image:**")
                    gr.Markdown("*Upload an image of the anime character you want to find*")
                    reference_char = gr.File(
                        label="Reference Character Image",
                        file_types=["image"]
                    )
                with gr.Column(scale=1):
                    similarity_threshold = gr.Slider(
                        minimum=0.3, maximum=0.9, value=0.6, step=0.05,
                        label="Character Similarity Threshold",
                        info="Lower = more matches (less strict), Higher = fewer matches (more strict)"
                    )
            
            with gr.Row():
                max_results = gr.Slider(
                    minimum=50, maximum=2000, value=500, step=50,
                    label="Maximum Results Per Video",
                    info="Limit the number of character matches per video"
                )
            
            with gr.Row():
                filter_btn = gr.Button("🔍 Find Character", variant="primary", size="lg")
                refresh_frames_btn = gr.Button("🔄 Refresh Available Frames", variant="secondary")
            
            character_status = gr.Markdown("Upload a character reference image to begin")
            
            # Character Recognition Results
            with gr.Row():
                character_gallery = gr.Gallery(
                    label="Character Matches (sorted by similarity)", 
                    columns=4, height=400
                )
            
            character_results = gr.JSON(label="Character Recognition Results")
        
        # Processing Summary
        with gr.Accordion("Processing Summary", open=False):
            summary_json = gr.JSON(label="Complete Processing Summary")
        
        # Functions
        def check_available_frames():
            """Check what frames are available from previous extractions"""
            try:
                available = get_available_extracted_frames(output_dir)
                if not available:
                    return "❌ No extracted frames found. Please run 'Anime Processing (Simple)' tab first to extract frames from videos."
                
                total_frames = sum(v["frame_count"] for v in available)
                video_list = ", ".join([v["video_name"] for v in available[:3]])
                if len(available) > 3:
                    video_list += f" and {len(available)-3} more"
                
                return f"✅ Found {len(available)} videos with {total_frames:,} extracted frames ready for character filtering\n📁 Videos: {video_list}"
            except Exception as e:
                return f"❌ Error checking available frames: {str(e)}"
        
        def filter_by_character(ref_img_path, threshold, max_res):
            """Character filtering on existing frames"""
            try:
                if not ref_img_path:
                    return [], {"error": "No reference character image provided"}, "❌ Please upload a reference character image"
                
                # Check if we have extracted frames
                available = get_available_extracted_frames(output_dir)
                if not available:
                    return [], {"error": "No extracted frames found"}, "❌ No extracted frames found. Please run 'Anime Processing (Simple)' tab first to extract frames from videos."
                
                # Only do character filtering, NO frame extraction
                results = process_character_filtering_batch(
                    frames_base_dir=output_dir,
                    reference_char_path=ref_img_path,
                    similarity_threshold=threshold,
                    max_results_per_video=max_res
                )
                
                if "error" in results:
                    return [], results, f"❌ Error: {results['error']}"
                
                # Get character matches for gallery
                char_name = results.get("character_name", "unknown")
                total_matches = results.get("total_character_matches", 0)
                
                gallery_items = []
                
                # Get preview images from character matches
                char_matches_base = os.path.join(output_dir, f"character_{char_name}_matches")
                if os.path.exists(char_matches_base):
                    try:
                        gallery_items = preview_character_matches(char_matches_base, max_preview=50)
                    except Exception as preview_error:
                        # If preview fails, try to get images directly
                        for root, dirs, files in os.walk(char_matches_base):
                            for file in files:
                                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    item_path = os.path.join(root, file)
                                    if os.path.exists(item_path):
                                        # Extract similarity score from filename if available
                                        try:
                                            parts = file.split('_')
                                            score = "Unknown"
                                            for part in parts:
                                                if '0.' in part and len(part) <= 5:
                                                    score = part
                                                    break
                                            gallery_items.append((item_path, f"Score: {score}"))
                                        except:
                                            gallery_items.append((item_path, f"Character match: {file}"))
                                    
                                    if len(gallery_items) >= 50:
                                        break
                            if len(gallery_items) >= 50:
                                break
                
                videos_processed = len(available)
                status_msg = f"✅ Character Search Complete! Found {total_matches} matches for character '{char_name}' across {videos_processed} videos"
                return gallery_items, results, status_msg
                
            except Exception as e:
                error_msg = f"❌ Error in character search: {str(e)}"
                return [], {"error": str(e)}, error_msg
        
        def refresh_available_frames():
            return check_available_frames()
        
        def create_summary(character_results):
            summary = {
                "character_recognition": character_results if character_results else {},
                "timestamp": datetime.now().isoformat()
            }
            return summary
        
        # Event handlers
        filter_btn.click(
            fn=filter_by_character,
            inputs=[reference_char, similarity_threshold, max_results],
            outputs=[character_gallery, character_results, character_status]
        )
        
        refresh_frames_btn.click(
            fn=refresh_available_frames,
            inputs=[],
            outputs=[available_status]
        )
        
        return global_status