import os
import time
import numpy as np
import cv2

print("Testing InsightFace Optional Dependencies\n" + "="*40)

# 1. Test Pillow
print("\n[1/6] Testing Pillow...")
try:
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='red')
    img.save('test_pillow.jpg')
    print("✅ Pillow is working correctly")
except Exception as e:
    print(f"❌ Pillow error: {e}")

# 2. Test matplotlib
print("\n[2/6] Testing matplotlib...")
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(2, 2))
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.savefig('test_matplotlib.jpg')
    plt.close()
    print("✅ matplotlib is working correctly")
except Exception as e:
    print(f"❌ matplotlib error: {e}")

# 3. Test scikit-learn
print("\n[3/6] Testing scikit-learn...")
try:
    from sklearn.cluster import KMeans
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print(f"✅ scikit-learn is working correctly (cluster centers: {kmeans.cluster_centers_})")
except Exception as e:
    print(f"❌ scikit-learn error: {e}")

# 4. Test scikit-image
print("\n[4/6] Testing scikit-image...")
try:
    from skimage import filters, io
    from skimage import img_as_ubyte  # Add this import
    
    # Create a simple image
    image = np.zeros((20, 20))
    image[5:15, 5:15] = 1
    
    # Apply a filter
    edges = filters.sobel(image)
    
    # Convert floating-point image to 8-bit for JPEG
    edges_8bit = img_as_ubyte(edges)
    io.imsave('test_skimage.jpg', edges_8bit)
    
    print("✅ scikit-image is working correctly")
except Exception as e:
    print(f"❌ scikit-image error: {e}")

# 5. Test dlib
print("\n[5/6] Testing dlib...")
try:
    import dlib
    detector = dlib.get_frontal_face_detector()
    print("✅ dlib is installed correctly")
    
    # For a more comprehensive test with face detection
    try:
        import insightface
        app = insightface.app.FaceAnalysis()
        app.prepare(ctx_id=0)
        
        # Get sample image with face
        test_img = cv2.imread("test_images/test_image_0.jpg")
        if test_img is None:
            # Create a simple face-like pattern
            test_img = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.ellipse(test_img, (150, 150), (100, 130), 0, 0, 360, (220, 180, 160), -1)
            cv2.circle(test_img, (100, 100), 25, (40, 40, 100), -1)
            cv2.circle(test_img, (200, 100), 25, (40, 40, 100), -1)
            cv2.ellipse(test_img, (150, 180), (50, 25), 0, 0, 360, (180, 100, 100), -1)
            cv2.imwrite("test_face.jpg", test_img)

        # Try dlib detector
        dets = detector(test_img)
        print(f"  - dlib detected {len(dets)} faces")
    except Exception as e:
        print(f"  - Could not run face detection test: {e}")
except Exception as e:
    print(f"❌ dlib error: {e}")

# 6. Test ONNX
print("\n[6/6] Testing ONNX...")
try:
    import onnx
    
    # Check if we can access the InsightFace models
    try:
        # Find path to InsightFace models
        import insightface
        app = insightface.app.FaceAnalysis()
        app.prepare(ctx_id=0)
        
        # Get model info
        model_files = []
        model_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
        
        if model_files:
            # Load one of the ONNX models
            model_path = os.path.join(model_dir, model_files[0])
            model = onnx.load(model_path)
            print(f"✅ ONNX is working correctly - loaded model: {model_files[0]}")
        else:
            print("✅ ONNX is installed correctly (no InsightFace models found to load)")
    except Exception as e:
        print(f"✅ ONNX is installed but couldn't test with InsightFace models: {e}")
except Exception as e:
    print(f"❌ ONNX error: {e}")

print("\n" + "="*40)
print("Test complete! Check for any ❌ errors above.")
print("Test files created in current directory.")
time.sleep(1)