import sys
import os
import shutil
import gradio as gr
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "front_end"))
from gradio_main import main_tab

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
    return [], "Uploads folder cleared!"

def restart_script():
    # Clear uploads before restarting
    clear_uploads()
    python = sys.executable
    os.execl(python, python, *sys.argv)

# Clean uploads folder on startup
clear_uploads()

with gr.Blocks() as app:
    file_input, preview_gallery, file_markdown, status = main_tab(
        upload_files, clear_uploads, restart_script
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)