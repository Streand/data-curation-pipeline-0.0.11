import insightface
import cv2
import requests
import os

# Download a sample image
img_url = "https://t4.ftcdn.net/jpg/02/18/93/97/360_F_218939757_YqHgeD3BAANU87y2Kc10Y40HNVgDv5rK.jpg"
img_path = os.path.join("image_path", "test_face.jpg")
with open(img_path, "wb") as f:
    f.write(requests.get(img_url).content)

img = cv2.imread(img_path)

# Use the new FaceAnalysis API
app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

faces = app.get(img)
print(f"Detected {len(faces)} faces")