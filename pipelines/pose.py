# pipelines/pose.py

def analyze_pose(image):
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
        return landmarks
    else:
        return None

def visualize_pose(image, landmarks):
    import cv2

    for landmark in landmarks:
        x, y = int(landmark[0] * image.shape[1]), int(landmark[1] * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    return image