import gradio as gr
import os
import cv2
from pipelines.face.detection import detect_faces

def list_uploaded_images(uploads_dir):
    if not os.path.exists(uploads_dir):
        return []
    return [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

def process_all_images(uploads_dir):
    image_files = list_uploaded_images(uploads_dir)
    if not image_files:
        return None, {"error": "No images found in uploads folder"}
    
    results = []
    gallery_images = []
    
    for filename in image_files:
        file_path = os.path.join(uploads_dir, filename)
        try:
            img_with_boxes, faces = detect_faces(file_path)
            img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            gallery_images.append((img_with_boxes, filename))
            results.append({
                "filename": filename,
                "faces": faces
            })
        except Exception as e:
            results.append({
                "filename": filename,
                "error": str(e)
            })
    
    return gallery_images, results

def face_tab(image_dir=None):
    with gr.Tab("Face Tab"):
        gr.Markdown("## Face Detection on Uploaded Images")

        if image_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            image_dir = os.path.join(base_dir, "uploads", "images")
    
        # Status display
        status = gr.Markdown("Click 'Run Face Detection' to process all uploaded images")
        
        # Run button
        run_btn = gr.Button("Run Face Detection")
        
        # Results
        gallery = gr.Gallery(label="Detected Faces", show_label=True, columns=3, height=500)
        results_json = gr.JSON(label="Face Detection Results")
        
        # Run detection on all images when button is clicked
        run_btn.click(
            fn=lambda: process_all_images(image_dir),  # Changed to image_dir
            inputs=None,
            outputs=[gallery, results_json]
        )