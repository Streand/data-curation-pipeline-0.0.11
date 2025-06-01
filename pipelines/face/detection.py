import cv2
import insightface

def detect_faces(image_path):
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(640, 640))
    img = cv2.imread(image_path)
    faces = model.get(img)
    for face in faces:
        bbox = face.bbox.tolist()
        score = face.det_score
        # Draw bounding box on the image (optional)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
    return img, [{"bbox": face.bbox.tolist(), "score": face.det_score} for face in faces]