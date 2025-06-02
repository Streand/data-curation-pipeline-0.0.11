import os

# GPU optimization settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import sys
import gradio as gr
import shutil
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "front_end"))
from front_end.UI_main import main_tab
from front_end.UI_accessories import accessories_tab
from front_end.UI_body import body_tab
from front_end.UI_camera import camera_tab
from front_end.UI_clothing import clothing_tab
from front_end.UI_face import face_tab
from front_end.UI_finalize import finalize_tab
from front_end.UI_nsfw import nsfw_tab
from front_end.UI_pose import pose_tab
from front_end.UI_video_stage_1 import video_tab_stage1

def upload_files(files):
    """Upload media files to the uploads directory.
    
    Args:
        files: List of file paths to upload
        
    Returns:
        str: Status message about the upload operation
    """
    import mimetypes
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "uploads")
    os.makedirs(save_dir, exist_ok=True)
    skipped = 0
    uploaded = 0
    
    for f in files:
        mime, _ = mimetypes.guess_type(f)
        if not mime or not (mime.startswith("image") or mime.startswith("video")):
            skipped += 1
            continue
            
        dest = os.path.join(save_dir, os.path.basename(f))
        if os.path.exists(dest):
            skipped += 1
            continue
        
        shutil.copy(f, dest)
        uploaded += 1
        
    if skipped > 0 and uploaded > 0:
        return f"Upload OK: {uploaded} file(s). Skipped: {skipped} file(s)."
    elif skipped > 0 and uploaded == 0:
        return f"No files uploaded. Skipped: {skipped} unsupported or duplicate file(s)."
    else:
        return f"Upload OK: {uploaded} file(s)."

def clear_uploads():
    """Remove all files from the uploads directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "uploads")
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception:
                pass  # Silently ignore errors during cleanup
    return "Uploads folder cleared!"

def restart_script():
    """Restart the application after clearing uploads."""
    clear_uploads()
    python = sys.executable
    os.execl(python, python, *sys.argv)

def update_previews(_=None):
    """Get list of media files and their names from the uploads folder.
    
    Returns:
        tuple: (list of media file paths, string of file names)
    """
    import mimetypes
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "uploads")
    media_files = []
    file_names = []
    if os.path.exists(save_dir):
        for f in os.listdir(save_dir):
            full_path = os.path.join(save_dir, f)
            mime, _ = mimetypes.guess_type(full_path)
            if mime and (mime.startswith("image") or mime.startswith("video")):
                media_files.append(full_path)
            file_names.append(f)
    return media_files, "\n".join(f"- {name}" for name in file_names)

# Clean uploads folder on startup
clear_uploads()

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")

with gr.Blocks() as app:
    main_tab(upload_files, clear_uploads, restart_script, update_previews)
    face_tab(UPLOADS_DIR)
    video_tab_stage1(UPLOADS_DIR)
    body_tab()
    pose_tab()
    camera_tab()
    clothing_tab()
    accessories_tab()
    nsfw_tab()
    finalize_tab()

if __name__ == "__main__":
    # Create required directories
    store_images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "store_images")
    os.makedirs(store_images_dir, exist_ok=True)
    
    video_stage1_dir = os.path.join(store_images_dir, "video_stage_1")
    os.makedirs(video_stage1_dir, exist_ok=True)
    
    # Launch the Gradio app with access to the store_images directory
    app.launch(
        inbrowser=True,
        allowed_paths=[store_images_dir]
    )