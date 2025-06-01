import sys
import os
import shutil
import gradio as gr

def upload_files(files):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "uploads")
    os.makedirs(save_dir, exist_ok=True)
    try:
        for f in files:
            dest = os.path.join(save_dir, os.path.basename(f))
            shutil.copy(f, dest)
        return "Upload OK"
    except Exception as e:
        return f"Upload failed: {e}"

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
    # Restart the Gradio app script
    python = sys.executable
    os.execl(python, python, *sys.argv)

def face_tab_fn():
    return "Face analysis results will appear here."

def clothing_tab_fn():
    return "Clothing analysis results will appear here."

def pose_tab_fn():
    return "Pose analysis results will appear here."

def body_tab_fn():
    return "Body analysis results will appear here."

def nsfw_tab_fn():
    return "NSFW analysis results will appear here."

def camera_tab_fn():
    return "Camera analysis results will appear here."

def accessories_tab_fn():
    return "Accessories analysis results will appear here."

def finish_tab_fn():
    return "Export and finish options will appear here."

with gr.Blocks() as app:
    with gr.Tab("Main"):
        gr.Markdown("### Upload Images and Videos")
        file_input = gr.File(file_count="multiple", type="filepath", label="Select images and/or video clips")
        status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            restart_btn = gr.Button("Restart", variant="stop", scale=0.5)
            clear_btn = gr.Button("Clear Uploads", variant="stop", scale=0.5)
            upload_btn = gr.Button("Upload", scale=1)
        preview_gallery = gr.Gallery(label="Image Preview", show_label=True, elem_id="preview_gallery", columns=4, height="auto")
        file_markdown = gr.Markdown(value="", label="All Uploaded Files")

        def update_previews(files):
            import mimetypes
            image_files = []
            file_names = []
            for f in files:
                mime, _ = mimetypes.guess_type(f)
                if mime and mime.startswith("image"):
                    image_files.append(f)
                file_names.append(os.path.basename(f))
            return image_files, "\n".join(f"- {name}" for name in file_names)

        upload_btn.click(
            upload_files, 
            inputs=[file_input], 
            outputs=[status]
        ).then(
            update_previews,
            inputs=[file_input],
            outputs=[preview_gallery, file_markdown]
        )
        clear_btn.click(
            clear_uploads,
            inputs=[],
            outputs=[status]
        ).then(
            lambda: ([], ""),
            inputs=[],
            outputs=[preview_gallery, file_markdown]
        )
        restart_btn.click(
            restart_script,
            inputs=[],
            outputs=[]
        )

    with gr.Tab("Face"):
        gr.Markdown("### Face Analysis")
        gr.Textbox(value=face_tab_fn(), interactive=False)

    with gr.Tab("Clothing"):
        gr.Markdown("### Clothing Analysis")
        gr.Textbox(value=clothing_tab_fn(), interactive=False)

    with gr.Tab("Pose"):
        gr.Markdown("### Pose Analysis")
        gr.Textbox(value=pose_tab_fn(), interactive=False)

    with gr.Tab("Body"):
        gr.Markdown("### Body Analysis")
        gr.Textbox(value=body_tab_fn(), interactive=False)

    with gr.Tab("NSFW"):
        gr.Markdown("### NSFW Analysis")
        gr.Textbox(value=nsfw_tab_fn(), interactive=False)

    with gr.Tab("Camera"):
        gr.Markdown("### Camera/Composition Analysis")
        gr.Textbox(value=camera_tab_fn(), interactive=False)

    with gr.Tab("Accessories"):
        gr.Markdown("### Accessories Analysis")
        gr.Textbox(value=accessories_tab_fn(), interactive=False)

    with gr.Tab("Finish"):
        gr.Markdown("### Finish & Export")
        gr.Textbox(value=finish_tab_fn(), interactive=False)

if __name__ == "__main__":
    app.launch(inbrowser=True)