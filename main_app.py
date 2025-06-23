# GPU optimization settings
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'       # See documentation: WORKFLOW.md -> # From Main_app.py -> GPU optimization settings
# os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'                  # See documentation: WORKFLOW.md -> # From Main_app.py -> GPU optimization settings
# os.environ['CUDA_MODULE_LOADING'] = 'LAZY'                            # See documentation: WORKFLOW.md -> # From Main_app.py -> GPU optimization settings
import os
import subprocess
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
from front_end.UI_video_tab import UI_video_tab


# Define constants for image and video subdirectories
IMAGES_SUBDIR = "images"
VIDEOS_SUBDIR = "videos"

def upload_files(files):
    """Upload media files to the uploads directory, splitting into images and videos subdirectories.
    
    Args:
        files: List of file paths to upload
        
    Returns:
        str: Status message about the upload operation
    """
    import mimetypes
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "uploads")
    images_dir = os.path.join(save_dir, IMAGES_SUBDIR)
    videos_dir = os.path.join(save_dir, VIDEOS_SUBDIR)
    
    # Create subdirectories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    skipped = 0
    uploaded = 0
    
    for f in files:
        mime, _ = mimetypes.guess_type(f)
        if not mime:
            skipped += 1
            continue
            
        # Determine the destination directory based on mime type
        if mime.startswith("image"):
            dest_dir = images_dir
        elif mime.startswith("video"):
            dest_dir = videos_dir
        else:
            skipped += 1
            continue
            
        dest = os.path.join(dest_dir, os.path.basename(f))
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
    """Remove all files from the uploads directory and its subdirectories."""
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
        
        # Create subdirectories after clearing
        os.makedirs(os.path.join(save_dir, IMAGES_SUBDIR), exist_ok=True)
        os.makedirs(os.path.join(save_dir, VIDEOS_SUBDIR), exist_ok=True)
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
    images_dir = os.path.join(save_dir, IMAGES_SUBDIR)
    videos_dir = os.path.join(save_dir, VIDEOS_SUBDIR)
    
    media_files = []
    file_names = []
    
    # Check images directory
    if os.path.exists(images_dir):
        for f in os.listdir(images_dir):
            full_path = os.path.join(images_dir, f)
            mime, _ = mimetypes.guess_type(full_path)
            if mime and mime.startswith("image"):
                media_files.append(full_path)
                file_names.append(f"[Image] {f}")
    
    # Check videos directory
    if os.path.exists(videos_dir):
        for f in os.listdir(videos_dir):
            full_path = os.path.join(videos_dir, f)
            mime, _ = mimetypes.guess_type(full_path)
            if mime and mime.startswith("video"):
                media_files.append(full_path)
                file_names.append(f"[Video] {f}")
    
    return media_files, "\n".join(f"- {name}" for name in file_names)

# Clean uploads folder on startup
clear_uploads()

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
UPLOADS_IMAGES_DIR = os.path.join(UPLOADS_DIR, IMAGES_SUBDIR)
UPLOADS_VIDEOS_DIR = os.path.join(UPLOADS_DIR, VIDEOS_SUBDIR)

def open_uploads_folder():
    """Open the uploads folder in Windows Explorer and bring it to the front."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "uploads")
    if sys.platform == "win32":
        # Open the folder directly without /select
        subprocess.Popen(f'explorer "{folder}"')
        return "Opened uploads folder."
    else:
        return "This function only works on Windows."

with gr.Blocks() as app:
    main_tab(upload_files, clear_uploads, restart_script, update_previews, open_uploads_folder)
    face_tab(UPLOADS_IMAGES_DIR)  # Pass images directory to face tab
    with gr.Tab("Video Processing"):
        UI_video_tab()  # Pass videos directory to video tab
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
    
    # Add new directories for anime and nsfw processing
    video_anime_dir = os.path.join(store_images_dir, "video_stage_anime")
    os.makedirs(video_anime_dir, exist_ok=True)
    
    video_nsfw_dir = os.path.join(store_images_dir, "video_stage_nsfw")
    os.makedirs(video_nsfw_dir, exist_ok=True)
    
    # Create image and video subdirectories in uploads
    os.makedirs(UPLOADS_IMAGES_DIR, exist_ok=True)
    os.makedirs(UPLOADS_VIDEOS_DIR, exist_ok=True)
    
    # Launch the Gradio app with access to the store_images directory
    app.launch(
        inbrowser=True,
        allowed_paths=[store_images_dir]
    )