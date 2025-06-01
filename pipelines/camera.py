import exifread
from PIL import Image

def analyze_camera_settings(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    
    camera_info = {
        "Make": tags.get("Image Make", "Unknown"),
        "Model": tags.get("Image Model", "Unknown"),
        "ExposureTime": tags.get("EXIF ExposureTime", "Unknown"),
        "FNumber": tags.get("EXIF FNumber", "Unknown"),
        "ISOSpeedRatings": tags.get("EXIF ISOSpeedRatings", "Unknown"),
        "DateTimeOriginal": tags.get("EXIF DateTimeOriginal", "Unknown"),
        "FocalLength": tags.get("EXIF FocalLength", "Unknown"),
    }
    
    return camera_info

def analyze_image(image_path):
    img = Image.open(image_path)
    camera_settings = analyze_camera_settings(image_path)
    return camera_settings, img.size  # Return camera settings and image dimensions