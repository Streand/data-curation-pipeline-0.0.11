def analyze_emotion(face_image):
    from deepface import DeepFace

    # Analyze the emotion of the given face image
    analysis = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
    
    # Extract the emotion with the highest confidence
    emotion = max(analysis[0]['emotion'], key=analysis[0]['emotion'].get)
    confidence = analysis[0]['emotion'][emotion]

    return emotion, confidence