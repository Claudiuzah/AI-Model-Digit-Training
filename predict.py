import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

def process_single_digit(img, debug=True):
    # Resize to 28x28 pixels
    img = cv2.resize(img, (28, 28))
    
    # Invert image if it has white background and black digits
    if img.mean() > 127:
        img = 255 - img
    
    if debug:
        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap='gray')
        plt.title('Processed Single Digit')
        plt.show()
    
    # Normalize to [0,1]
    img = img / 255.0
    
    # Reshape for model input
    img = img.reshape(1, 28, 28, 1)
    
    return img

def process_multiple_digits(img, debug=True):
    # Find contours to identify individual digits
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    digit_contours = []
    for contour in contours:
        # Filter out very small contours
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            digit_contours.append((x, y, w, h, contour))
    
    # Sort by x coordinate (left to right)
    digit_contours.sort(key=lambda x: x[0])
    
    processed_digits = []
    original_positions = []
    
    for x, y, w, h, contour in digit_contours:
        # Add padding around the digit
        padding = int(max(w, h) * 0.3)
        img_height, img_width = img.shape
        pad_x = max(0, x - padding)
        pad_y = max(0, y - padding)
        pad_w = min(img_width - pad_x, w + 2*padding)
        pad_h = min(img_height - pad_y, h + 2*padding)
        
        # Extract and process the digit
        digit = img[pad_y:pad_y+pad_h, pad_x:pad_x+pad_w]
        digit = cv2.resize(digit, (28, 28))
        
        if digit.mean() > 127:
            digit = 255 - digit
        
        # Store original position
        original_positions.append((x, y, w, h))
        
        # Normalize and reshape
        digit = digit / 255.0
        digit = digit.reshape(1, 28, 28, 1)
        processed_digits.append(digit)
    
    if debug and processed_digits:
        plt.figure(figsize=(len(processed_digits) * 2, 2))
        for i, digit in enumerate(processed_digits):
            plt.subplot(1, len(processed_digits), i + 1)
            plt.imshow(digit.reshape(28, 28), cmap='gray')
            plt.title(f'Digit {i+1}')
        plt.show()
    
    return processed_digits, original_positions

def preprocess_image(image_path, debug=True, multi_digit=False):
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
    
    if not multi_digit:
        # Single digit mode
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
            return process_single_digit(img, debug)
    else:
        # Multi-digit mode
        return process_multiple_digits(img, debug)
    
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

def predict_digit(image_path, multi_digit=False):
    # Preprocess the image
    if not multi_digit:
        processed_image = preprocess_image(image_path, multi_digit=False)
        
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
    else:
        processed_digits, positions = preprocess_image(image_path, multi_digit=True)
        
        if not processed_digits:
            return []
        
        results = []
        for digit_img, pos in zip(processed_digits, positions):
            predictions = model.predict(digit_img, verbose=0)
            digit = np.argmax(predictions[0])
            confidence = predictions[0][digit] * 100
            x, y, w, h = pos
            results.append({
                'digit': digit,
                'confidence': confidence,
                'position': {'x': x, 'y': y, 'width': w, 'height': h}
            })
        
        return results

# Example usage
# image_path = 'cifra_0.png' 
# image_path = 'cifra_2.png'
# image_path = 'cifra_4.png'
# image_path = 'cifra_5.png'
image_path = 'cifra_9.png'

try:
    predictions = predict_digit(image_path)
    print("\nTop 3 predictions:")
    for digit, confidence in predictions:
        print(f"Digit {digit}: {confidence:.2f}% confidence")
except Exception as e:
    print(f"Error processing image: {e}")
