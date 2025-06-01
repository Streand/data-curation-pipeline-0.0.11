import sys
import os
import shutil
import gradio as gr
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "front_end"))
from front_end.UI_main import main_tab

def upload_files(files):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "uploads")
    os.makedirs(save_dir, exist_ok=True)
    skipped = 0
    for f in files:
        dest = os.path.join(save_dir, os.path.basename(f))
        if os.path.exists(dest):
            skipped += 1
            continue
        shutil.copy(f, dest)
    if skipped > 0:
        return f"Upload OK. Skipped: {skipped} file(s)"
    return "Upload OK"

def clear_uploads():
    # Clear the 'uploads' folder by deleting all its contents
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
            except Exception as e:
                pass  # Optionally, handle/log errors
    return "Uploads folder cleared!"

def restart_script():
    # Clear uploads before restarting
    clear_uploads()
    python = sys.executable
    os.execl(python, python, *sys.argv)

def update_previews(_=None):
    import mimetypes
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "uploads")
    image_files = []
    file_names = []
    if os.path.exists(save_dir):
        for f in os.listdir(save_dir):
            full_path = os.path.join(save_dir, f)
            mime, _ = mimetypes.guess_type(full_path)
            if mime and mime.startswith("image"):
                image_files.append(full_path)
            file_names.append(f)
    return image_files, "\n".join(f"- {name}" for name in file_names)

# Clean uploads folder on startup
clear_uploads()

with gr.Blocks() as app:
    file_input, preview_gallery, status = main_tab(
        upload_files, clear_uploads, restart_script, update_previews
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)