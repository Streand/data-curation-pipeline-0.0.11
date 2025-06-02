import insightface
import cv2

# Download a sample image
img_url = "https://raw.githubusercontent.com/deepinsight/insightface/master/images/t1.jpg"
img_path = "test_face.jpg"

import requests
with open(img_path, "wb") as f:
    f.write(requests.get(img_url).content)

# Load the image
img = cv2.imread(img_path)

# Load the face detector (uses ONNX and onnxruntime-gpu by default)
detector = insightface.model_zoo.get_model('buffalo_l')
detector.prepare(ctx_id=0)  # 0 = first GPU

faces = detector.detect(img)
print(f"Detected {len(faces)} faces")