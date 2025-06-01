def extract_features(image, face_data):
    import cv2
    import numpy as np

    features = []
    
    for face in face_data:
        # Extract bounding box
        bbox = face['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop the face from the image
        face_image = image[y1:y2, x1:x2]
        
        # Example feature extraction: color histogram
        hist = cv2.calcHist([face_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        features.append({
            'bbox': bbox,
            'color_histogram': hist.tolist()
        })
    
    return features