import gradio as gr
from pipelines.controller import process_input

def main():
    with gr.Blocks(title="Data Curation Pipeline") as app:
        gr.Markdown("# Data Curation Pipeline")
        with gr.Tab("Input"):
            input_image = gr.Image(label="Upload Image", type="filepath")
            input_video = gr.Video(label="Upload Video")
            process_btn = gr.Button("Process")
        with gr.Tab("Pipeline Configuration"):
            face_analysis = gr.Checkbox(label="Face Analysis", value=True)
            clothing_analysis = gr.Checkbox(label="Clothing Analysis", value=True)
            pose_analysis = gr.Checkbox(label="Pose Analysis", value=True)
            nsfw_analysis = gr.Checkbox(label="NSFW Analysis", value=True)
        with gr.Tab("Results"):
            output_image = gr.Image(label="Processed Image")
            output_json = gr.JSON(label="Extracted Metadata")
            export_btn = gr.Button("Export Results")
        process_btn.click(
            process_input, 
            inputs=[input_image, input_video, face_analysis, clothing_analysis, pose_analysis, nsfw_analysis],
            outputs=[output_image, output_json]
        )
    app.launch()

if __name__ == "__main__":
    main()