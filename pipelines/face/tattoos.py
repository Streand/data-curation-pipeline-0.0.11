def analyze_tattoos(image_path):
    import cv2
    import numpy as np
    from ultralytics import YOLO

    # Load the YOLO model for tattoo detection
    model = YOLO('path/to/tattoo_detection_model.pt')

    # Read the image
    img = cv2.imread(image_path)

    # Perform tattoo detection
    results = model(img)

    # Process results
    detections = []
    for result in results:
        for detection in result.boxes:
            bbox = detection.xyxy[0].tolist()  # Get bounding box coordinates
            confidence = detection.conf[0].item()  # Get confidence score
            detections.append({"bbox": bbox, "confidence": confidence})

    return img, detections