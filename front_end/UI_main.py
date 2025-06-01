import gradio as gr
import os
import base64

def generate_preview_html(image_paths):
    # Generates HTML for image previews using base64 encoding
    html = '<div style="display: flex; flex-wrap: wrap; gap: 10px;">'
    for path in image_paths:
        try:
            with open(path, "rb") as f:
                img_data = f.read()
            img_b64 = base64.b64encode(img_data).decode('utf-8')
            ext = os.path.splitext(path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            html += f'<img src="data:{mime};base64,{img_b64}" style="height:250px; width:auto; object-fit:contain; border:1px solid #ccc; border-radius:4px; background:#222;">'
        except Exception:
            html += '<div style="width:40px; height:40px; border:1px solid red; color:#fff; display:flex; align-items:center; justify-content:center;">Error</div>'
    html += '</div>'
    return html

def main_tab(upload_files, clear_uploads, restart_script, update_previews):
    with gr.Tab("Main"):
        gr.Markdown("### Upload Images and Videos")
        file_input = gr.File(file_count="multiple", type="filepath", label="Select images and/or video clips")
        status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            restart_btn = gr.Button("Restart", variant="stop", scale=1)
            clear_btn = gr.Button("Clear Uploads", variant="stop", scale=1)
            upload_btn = gr.Button("Upload", variant="primary", scale=2)
        preview_html = gr.HTML(label="Image Preview")

        def update_preview_html():
            images, _ = update_previews()
            return generate_preview_html(images)

        upload_btn.click(
            upload_files, 
            inputs=[file_input], 
            outputs=[status]
        ).then(
            update_preview_html,
            inputs=[],
            outputs=[preview_html]
        )
        clear_btn.click(
            clear_uploads,
            inputs=[],
            outputs=[status]
        ).then(
            update_preview_html,
            inputs=[],
            outputs=[preview_html]
        )
        restart_btn.click(
            restart_script,
            inputs=[],
            outputs=[]
        )

        # --- Set initial preview on app startup ---
        initial_images, _ = update_previews()
        preview_html.value = generate_preview_html(initial_images)
        # ------------------------------------------

    return file_input, preview_html, status