import gradio as gr

def main_tab(upload_files, clear_uploads, restart_script, update_previews):
    with gr.Tab("Main"):
        gr.Markdown("### Upload Images and Videos")
        file_input = gr.File(file_count="multiple", type="filepath", label="Select images and/or video clips")
        status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            restart_btn = gr.Button("Restart", variant="stop", scale=1)
            clear_btn = gr.Button("Clear Uploads", variant="stop", scale=1)
            upload_btn = gr.Button("Upload", variant="primary", scale=2)
        preview_gallery = gr.Gallery(label="Image Preview", show_label=True, elem_id="preview_gallery", columns=4, height="auto")

        upload_btn.click(
            upload_files, 
            inputs=[file_input], 
            outputs=[status]
        ).then(
            lambda: update_previews()[0],
            inputs=[],
            outputs=[preview_gallery]
        )
        clear_btn.click(
            clear_uploads,
            inputs=[],
            outputs=[status]
        ).then(
            lambda: update_previews()[0],
            inputs=[],
            outputs=[preview_gallery]
        )
        restart_btn.click(
            restart_script,
            inputs=[],
            outputs=[]
        )

        # --- Set initial preview on app startup ---
        initial_images, _ = update_previews()
        preview_gallery.value = initial_images
        # ------------------------------------------

    return file_input, preview_gallery, status