import os
import gradio as gr
import sys
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines"))
from pipelines.videostageallframes import process_video_all_frames, process_videos_batch_all_frames, check_gpu_optimization

def open_folder(folder_path):
    """Open the specified folder in Windows File Explorer"""
    if not folder_path:
        return "No folder specified"
    
    if not os.path.exists(folder_path):
        return f"Folder not found: {folder_path}"
    
    try:
        import subprocess
        subprocess.run(['explorer', '/select,', folder_path], shell=True)
        return f"Opened folder: {folder_path}"
    except Exception as e:
        return f"Error opening folder: {str(e)}"

def UI_video_stage_allframes(video_dir=None):
    with gr.Tab("All Frames Extraction"):
        gr.Markdown("## All Frames Extraction")
        gr.Markdown("### Extract every single frame from videos with GPU acceleration")
        
        # Default directories
        if video_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_dir = os.path.join(base_dir, "uploads", "videos")
        
        # Output directory (as specified)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "all_data_storage", "allframes")
        os.makedirs(output_dir, exist_ok=True)
        
        # GPU Status
        gpu_info = check_gpu_optimization()
        gpu_status = "🚀 **GPU Status:** "
        if gpu_info["has_cuda"]:
            gpu_status += f"{gpu_info['device_name']} | {gpu_info['memory_gb']} GB VRAM"
            if gpu_info["is_blackwell"]:
                gpu_status += f" | **Blackwell Architecture Detected** - Optimized batch size: {gpu_info['recommended_batch']}"
        else:
            gpu_status += "No CUDA GPU detected - will use CPU (slower)"
        
        gr.Markdown(gpu_status)
        
        # Processing Mode Selection
        with gr.Row():
            processing_mode = gr.Radio(
                choices=["Single Video", "Batch Process Directory"],
                label="Processing Mode",
                value="Single Video"
            )
        
        # Single Video Mode
        with gr.Group(visible=True) as single_video_group:
            gr.Markdown("### Single Video Processing")
            single_video_file = gr.File(
                label="Select Video File",
                file_types=[".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]
            )
            
            with gr.Row():
                process_single_btn = gr.Button("🚀 Extract All Frames", variant="primary", size="lg")
                open_single_output_btn = gr.Button("📁 Open Output Folder", variant="secondary")
        
        # Batch Processing Mode
        with gr.Group(visible=False) as batch_group:
            gr.Markdown("### Batch Directory Processing")
            gr.Markdown(f"**Video Directory:** `{video_dir}`")
            
            def check_video_directory():
                if not os.path.exists(video_dir):
                    return f"❌ Directory not found: {video_dir}"
                
                video_files = [f for f in os.listdir(video_dir) 
                              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'))]
                
                if not video_files:
                    return f"❌ No video files found in: {video_dir}"
                
                total_size = 0
                for video_file in video_files[:5]:  # Check first 5 files for size estimate
                    try:
                        file_path = os.path.join(video_dir, video_file)
                        total_size += os.path.getsize(file_path)
                    except:
                        pass
                
                avg_size_mb = (total_size / len(video_files[:5])) / (1024*1024) if video_files else 0
                estimated_total_mb = avg_size_mb * len(video_files)
                
                status = f"✅ Found {len(video_files)} videos\n"
                status += f"📁 Average size: {avg_size_mb:.1f} MB per video\n"
                status += f"📊 Estimated total: {estimated_total_mb:.1f} MB\n"
                status += f"📋 Sample files: {', '.join(video_files[:3])}"
                if len(video_files) > 3:
                    status += f" and {len(video_files)-3} more..."
                
                return status
            
            directory_status = gr.Markdown(check_video_directory())
            
            with gr.Row():
                refresh_dir_btn = gr.Button("🔄 Refresh Directory", variant="secondary")
                process_batch_btn = gr.Button("🚀 Process All Videos", variant="primary", size="lg")
                open_batch_output_btn = gr.Button("📁 Open Output Folder", variant="secondary")
        
        # Progress and Status
        progress_bar = gr.Progress()
        status_display = gr.Markdown("Ready to extract frames")
        
        # Results Display
        with gr.Accordion("Processing Results", open=False):
            results_json = gr.JSON(label="Detailed Results")
        
        # Preview Gallery (limited to prevent UI slowdown)
        with gr.Accordion("Sample Frames Preview", open=False):
            gr.Markdown("*Preview limited to 20 frames for performance*")
            preview_gallery = gr.Gallery(
                label="Sample Extracted Frames",
                columns=5,
                height=300,
                object_fit="cover"
            )
        
        # Functions
        def toggle_processing_mode(mode):
            if mode == "Single Video":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        def process_single_video(video_file, progress=gr.Progress()):
            if not video_file:
                return "❌ Please select a video file", {}, []
            
            try:
                start_time = time.time()
                
                # Progress callback
                def progress_callback(prog, message):
                    progress(prog, message)
                
                result = process_video_all_frames(video_file.name, output_dir, progress_callback)
                
                if "error" in result:
                    return f"❌ Error: {result['error']}", result, []
                
                # Success message
                elapsed = time.time() - start_time
                frames_count = result.get("frames_extracted", 0)
                video_name = result.get("video_name", "unknown")
                
                status_msg = f"✅ **Extraction Complete!**\n"
                status_msg += f"📹 Video: {video_name}\n"
                status_msg += f"🖼️ Frames extracted: {frames_count:,}\n"
                status_msg += f"⏱️ Processing time: {elapsed:.2f} seconds\n"
                status_msg += f"📁 Output: {result.get('output_directory', 'Unknown')}"
                
                # Create preview gallery
                preview_images = []
                output_folder = result.get('output_directory')
                if output_folder and os.path.exists(output_folder):
                    frame_files = [f for f in os.listdir(output_folder) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    frame_files.sort()
                    
                    # Take every Nth frame for preview (max 20)
                    step = max(1, len(frame_files) // 20)
                    for i in range(0, len(frame_files), step):
                        if len(preview_images) >= 20:
                            break
                        frame_path = os.path.join(output_folder, frame_files[i])
                        preview_images.append((frame_path, f"Frame {i}"))
                
                return status_msg, result, preview_images
                
            except Exception as e:
                error_msg = f"❌ Processing failed: {str(e)}"
                return error_msg, {"error": str(e)}, []
        
        def process_batch_videos(progress=gr.Progress()):
            try:
                start_time = time.time()
                
                # Progress callback
                def progress_callback(prog, message):
                    progress(prog, message)
                
                result = process_videos_batch_all_frames(video_dir, output_dir, progress_callback)
                
                if "error" in result:
                    return f"❌ Error: {result['error']}", result, []
                
                # Success message
                elapsed = time.time() - start_time
                total_frames = result.get("total_frames_extracted", 0)
                videos_processed = result.get("total_videos_processed", 0)
                successful = result.get("successful_videos", 0)
                
                status_msg = f"✅ **Batch Processing Complete!**\n"
                status_msg += f"📹 Videos processed: {successful}/{videos_processed}\n"
                status_msg += f"🖼️ Total frames extracted: {total_frames:,}\n"
                status_msg += f"⏱️ Total processing time: {elapsed:.2f} seconds\n"
                status_msg += f"📁 Output directory: {output_dir}"
                
                if result.get("gpu_info", {}).get("has_cuda"):
                    gpu_name = result["gpu_info"].get("device_name", "Unknown GPU")
                    status_msg += f"\n🚀 GPU acceleration: {gpu_name}"
                
                # Create preview gallery from multiple videos
                preview_images = []
                for video_result in result.get("results", [])[:3]:  # First 3 videos
                    if video_result.get("status") == "success":
                        video_output_dir = video_result.get("output_directory")
                        if video_output_dir and os.path.exists(video_output_dir):
                            frame_files = [f for f in os.listdir(video_output_dir) 
                                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            frame_files.sort()
                            
                            # Take a few frames from each video
                            step = max(1, len(frame_files) // 5)
                            video_name = video_result.get("video_name", "unknown")
                            
                            for i in range(0, min(len(frame_files), step * 5), step):
                                if len(preview_images) >= 20:
                                    break
                                frame_path = os.path.join(video_output_dir, frame_files[i])
                                preview_images.append((frame_path, f"{video_name} - Frame {i}"))
                
                return status_msg, result, preview_images
                
            except Exception as e:
                error_msg = f"❌ Batch processing failed: {str(e)}"
                return error_msg, {"error": str(e)}, []
        
        def refresh_directory():
            return check_video_directory()
        
        def open_output_folder():
            return open_folder(output_dir)
        
        # Event Handlers
        processing_mode.change(
            fn=toggle_processing_mode,
            inputs=[processing_mode],
            outputs=[single_video_group, batch_group]
        )
        
        process_single_btn.click(
            fn=process_single_video,
            inputs=[single_video_file],
            outputs=[status_display, results_json, preview_gallery]
        )
        
        process_batch_btn.click(
            fn=process_batch_videos,
            inputs=[],
            outputs=[status_display, results_json, preview_gallery]
        )
        
        refresh_dir_btn.click(
            fn=refresh_directory,
            inputs=[],
            outputs=[directory_status]
        )
        
        open_single_output_btn.click(
            fn=open_output_folder,
            inputs=[],
            outputs=[status_display]
        )
        
        open_batch_output_btn.click(
            fn=open_output_folder,
            inputs=[],
            outputs=[status_display]
        )
        
        return status_display