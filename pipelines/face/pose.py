def analyze_pose(image_path):
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            h, w, _ = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

    return image, results.pose_landmarks.landmark if results.pose_landmarks else None