import os
import cv2

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image file {image_path} does not exist.")
    image = cv2.imread(image_path)
    return image

def load_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file {video_path} does not exist.")
    video_capture = cv2.VideoCapture(video_path)
    return video_capture

def extract_frames(video_capture, frame_rate=1):
    frames = []
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        current_frame = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame % frame_interval == 0:
            frames.append(frame)
    
    video_capture.release()
    return frames