def analyze_nsfw_content(image_path):
    import cv2
    import opennsfw
    model = opennsfw.NSFWModel()
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    # Preprocess the image for NSFW analysis
    processed_image = preprocess_image(image)
    
    # Analyze the image for NSFW content
    nsfw_score = model.predict(processed_image)
    
    # Return the NSFW score and a classification
    classification = "NSFW" if nsfw_score > 0.5 else "SFW"
    return nsfw_score, classification

def preprocess_image(image):
    # Resize and normalize the image for the NSFW model
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize to [0, 1]
    return image

def nsfw_analysis(image_path):
    nsfw_score, classification = analyze_nsfw_content(image_path)
    return {"score": nsfw_score, "classification": classification}