def analyze_accessories(image_path):
    import cv2
    import numpy as np
    from ultralytics import YOLO

    # Load the YOLO model for accessory detection
    model = YOLO('path/to/yolo/model')  # Specify the path to your YOLO model

    # Read the image
    img = cv2.imread(image_path)
    results = model(img)

    accessories = []
    for result in results:
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = detection
            accessories.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class": cls
            })

    return img, accessories