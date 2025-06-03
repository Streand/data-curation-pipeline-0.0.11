import os
import gradio as gr


#  in UI_video_tab.py, you're passing the video directory:
    #-------------code:  UI_video_stage_1(video_dir)

def UI_video_stage_1(video_dir=None):
    with gr.Tab("Video Stage 1"):
        gr.Markdown("## Video Stage 1 - OpenCV, InsightFace")

        # Default directory if none provided
        if video_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_dir = os.path.join(base_dir, "uploads", "videos")
        
        # Now you can use video_dir to access videos
        # For example, you could list available videos:
        video_files = []
        if os.path.exists(video_dir):
            video_files = [f for f in os.listdir(video_dir) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        # Display available videos
        if video_files:
            gr.Markdown(f"Found {len(video_files)} videos in directory")
            video_dropdown = gr.Dropdown(
                choices=video_files,
                label="Select a video to process"
            )
        else:
            gr.Markdown("No videos found. Please upload videos first.")
            
###### Add your UI components here that use video_dir #########################################################################################################