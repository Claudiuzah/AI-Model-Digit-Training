import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

def preprocess_image(image_path, debug=True):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read the image")
    
    # Store original for display
    original = img.copy()
    
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                              cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours to center the digit
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the main contour (the digit)
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Add padding around the digit
        padding = int(max(w, h) * 0.3)  # Dynamic padding based on digit size
        img_height, img_width = img.shape
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_width - x, w + 2*padding)
        h = min(img_height - y, h + 2*padding)
        
        # Crop to the digit with padding
        img = img[y:y+h, x:x+w]
    
    # Resize to 28x28 pixels
    img = cv2.resize(img, (28, 28))
    
    # Invert image if it has white background and black digits
    if img.mean() > 127:
        img = 255 - img
    
    if debug:
        plt.figure(figsize=(8, 2))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
        plt.title('Original Image')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img, cmap='gray')
        plt.title('Preprocessed Image')
        
        plt.show()
    
    # Normalize to [0,1]
    img = img / 255.0
    
    # Reshape for model input
    img = img.reshape(1, 28, 28, 1)
    
    return img

def predict_digit(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    
    # Get top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    results = []
    for idx in top_3_indices:
        digit = idx
        confidence = predictions[0][idx] * 100
        results.append((digit, confidence))
    
    return results

# Example usage
# image_path = 'cifra_0.png' # Replace with your image path 
image_path = 'cifra_2.png'
# image_path = 'cifra_4.png'
# image_path = 'cifra_5.png'
# image_path = 'cifra_9.png'

try:
    predictions = predict_digit(image_path)
    print("\nTop 3 predictions:")
    for digit, confidence in predictions:
        print(f"Digit {digit}: {confidence:.2f}% confidence")
except Exception as e:
    print(f"Error processing image: {e}")
