
import gradio as gr
import os
from front_end.UI_video_stage_1 import UI_video_stage_1
from front_end.UI_video_stage_2 import UI_video_stage_2
from front_end.UI_video_stage_3 import UI_video_stage_3

def UI_video_tab(video_dir=None):
    with gr.Tab("Video Tab"):
        gr.Markdown("## Video")
        
        if video_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_dir = os.path.join(base_dir, "uploads", "videos")
        
        # Now you can pass video_dir to your video stage components if needed
        UI_video_stage_1(video_dir)
        UI_video_stage_2()
        UI_video_stage_3()
        
        # Add your UI components here that use video_dir