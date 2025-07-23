import cv2
import numpy as np
import tensorflow as tf
from predict import predict_digit

# Load the model
model = tf.keras.models.load_model('mnist_model.h5')

def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Save frame temporarily
    cv2.imwrite('temp_frame.png', gray)
    
    # Get predictions
    try:
        results = predict_digit('temp_frame.png', multi_digit=True)
        
        # Draw predictions on frame
        for result in results:
            digit = result['digit']
            confidence = result['confidence']
            pos = result['position']
            
            # Draw bounding box
            cv2.rectangle(frame, 
                         (pos['x'], pos['y']), 
                         (pos['x'] + pos['width'], pos['y'] + pos['height']), 
                         (0, 255, 0), 2)
            
            # Draw prediction
            text = f"{digit} ({confidence:.1f}%)"
            cv2.putText(frame, text, 
                       (pos['x'], pos['y'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0), 2)
    except Exception as e:
        print(f"Error processing frame: {e}")
    
    return frame

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame
        processed_frame = process_frame(frame)
        
        # Display the frame
        cv2.imshow('Digit Recognition', processed_frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
