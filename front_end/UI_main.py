import gradio as gr
import os
import base64

def generate_preview_html(image_paths):
    html = """
    <style>
    .img-thumb { cursor:pointer; transition:box-shadow 0.2s; }
    .img-thumb:hover { box-shadow:0 0 8px #fff; }
    .img-overlay-bg {
        display:none; position:fixed; z-index:9999; left:0; top:0; width:100vw; height:100vh;
        background:rgba(0,0,0,0.85); align-items:center; justify-content:center;
    }
    .img-overlay-bg.active { display:flex; }
    .img-overlay-content {
        position:relative; background:transparent; padding:0; border-radius:8px;
        max-width:90vw; max-height:90vh; display:flex; align-items:center; justify-content:center;
    }
    .img-overlay-content img { 
        max-width:80vw; 
        max-height:80vh; 
        border-radius:8px; 
        box-shadow:0 0 24px #000; 
    }
    .img-overlay-close {
        position:absolute; top:-32px; right:-32px; color:#fff; background:#222; border-radius:50%; width:32px; height:32px;
        display:flex; align-items:center; justify-content:center; font-size:22px; cursor:pointer; border:2px solid #fff;
        z-index:10001;
    }
    @media (max-width:600px) {
        .img-overlay-content img { max-width:98vw; max-height:60vh; }
    }
    </style>
    <div id="img-overlay-bg" class="img-overlay-bg" onclick="this.classList.remove('active');">
        <div class="img-overlay-content" onclick="event.stopPropagation();">
            <span class="img-overlay-close" onclick="document.getElementById('img-overlay-bg').classList.remove('active');">&times;</span>
            <img id="img-overlay-img" src="" />
        </div>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
    """
    
    # Generate thumbnails with INLINE onclick handlers (this is the key change)
    for idx, path in enumerate(image_paths):
        try:
            with open(path, "rb") as f:
                img_data = f.read()
            img_b64 = base64.b64encode(img_data).decode('utf-8')
            ext = os.path.splitext(path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            thumb = f"data:{mime};base64,{img_b64}"
            
            # Direct inline handler - no JavaScript needed
            html += f'''<img 
                class="img-thumb" 
                src="{thumb}" 
                onclick="document.getElementById('img-overlay-img').src='{thumb}';document.getElementById('img-overlay-bg').classList.add('active');event.stopPropagation();" 
                style="height:120px; width:auto; object-fit:contain; border:1px solid #ccc; border-radius:4px; background:#222;">'''
                
        except Exception:
            html += '<div style="width:120px; height:120px; border:1px solid red; color:#fff; display:flex; align-items:center; justify-content:center;">Error</div>'
    
    html += "</div>"
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