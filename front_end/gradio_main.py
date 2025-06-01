import gradio as gr

def main_tab(upload_files, clear_uploads, restart_script, update_previews):
    with gr.Tab("Main"):
        gr.Markdown("### Upload Images and Videos")
        file_input = gr.File(file_count="multiple", type="filepath", label="Select images and/or video clips")
        status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            restart_btn = gr.Button("Restart", variant="stop", scale=0.25)
            clear_btn = gr.Button("Clear Uploads", variant="stop", scale=0.25)
            upload_btn = gr.Button("Upload", variant="primary", scale=1)
        preview_gallery = gr.Gallery(label="Image Preview", show_label=True, elem_id="preview_gallery", columns=4, height="auto")
        file_markdown = gr.Markdown(value="", label="All Uploaded Files")

        upload_btn.click(
            upload_files, 
            inputs=[file_input], 
            outputs=[status]
        ).then(
            update_previews,
            inputs=[],
            outputs=[preview_gallery, file_markdown]
        )
        clear_btn.click(
            clear_uploads,
            inputs=[],
            outputs=[status]
        ).then(
            update_previews,
            inputs=[],
            outputs=[preview_gallery, file_markdown]
        )
        restart_btn.click(
            restart_script,
            inputs=[],
            outputs=[]
        )

        # --- Set initial preview and file list on app startup ---
        initial_images, initial_list = update_previews()
        preview_gallery.value = initial_images
        file_markdown.value = initial_list
        # -------------------------------------------------------

    return file_input, preview_gallery, file_markdown, status