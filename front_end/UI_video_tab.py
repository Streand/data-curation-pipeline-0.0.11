import gradio as gr
import os
from front_end.UI_video_stage_1 import UI_video_stage_1
from front_end.UI_video_stage_2 import UI_video_stage_2
from front_end.UI_video_stage_anime import UI_video_stage_anime
from front_end.UI_video_stage_anime2 import UI_video_stage_anime2
from front_end.UI_video_stage_nsfw import UI_video_stage_nsfw

def UI_video_tab(video_dir=None):
    with gr.Tab("Video Processing"):
        gr.Markdown("## Video Frame Extraction & Analysis")
        
        if video_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_dir = os.path.join(base_dir, "uploads", "videos")
        
        # Create tabs for different video processing stages
        with gr.Tabs():
            # Stage 1: Basic video processing (faces, etc.)
            with gr.TabItem("Stage 1 - Face Detection"):
                UI_video_stage_1(video_dir)
            
            # Stage 2: Additional processing
            with gr.TabItem("Stage 2"):
                UI_video_stage_2()
                
            # Original anime processing
            with gr.TabItem("Anime Processing (Simple)"):
                UI_video_stage_anime(video_dir)
            
            # New two-stage anime processing
            with gr.TabItem("Anime Processing (2-Stage)"):
                UI_video_stage_anime2(video_dir)
            
            # NSFW analysis
            with gr.TabItem("NSFW Analysis"):
                UI_video_stage_nsfw(video_dir)