import streamlit as st
import tensorflow as tf
import numpy as np
from predict import predict_digit, preprocess_image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2
from PIL import Image
import io

# Set page config
st.set_page_config(page_title="Digit Predictor", layout="wide")
st.title("Handwritten Digit Recognition")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_model.h5')

model = load_model()

# Function to generate confusion matrix
@st.cache_data
def generate_confusion_matrix():
    # Load test data
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Get predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    return cm

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Image Upload", "Webcam", "Model Performance"])

with tab1:
    st.header("Upload and Predict")
    multi_digit = st.checkbox("Enable multi-digit recognition")
    uploaded = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])
    
    if uploaded:
        # Save uploaded file
        with open("temp_img.png", "wb") as f:
            f.write(uploaded.read())
        
        # Make prediction
        try:
            preds = predict_digit("temp_img.png", multi_digit=multi_digit)
            
            # Display image
            img = cv2.imread("temp_img.png")
            
            if multi_digit:
                # Draw predictions on image for multi-digit mode
                for result in preds:
                    digit = result['digit']
                    confidence = result['confidence']
                    pos = result['position']
                    
                    # Draw bounding box
                    cv2.rectangle(img, 
                                (pos['x'], pos['y']), 
                                (pos['x'] + pos['width'], pos['y'] + pos['height']), 
                                (0, 255, 0), 2)
                    
                    # Draw prediction
                    cv2.putText(img, f"{digit} ({confidence:.1f}%)", 
                              (pos['x'], pos['y'] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              (0, 255, 0), 2)
                
                st.image(img, channels="BGR", caption="Predictions", use_column_width=True)
                
                # Display results
                st.subheader("Detected Digits (left to right):")
                result_text = " ".join([str(r['digit']) for r in preds])
                st.write(f"Complete number: {result_text}")
                
            else:
                # Single digit mode
                st.image("temp_img.png", caption="Uploaded Image", use_column_width=True)
                st.subheader("Predictions:")
                for digit, confidence in preds:
                    st.write(f"Digit {digit}: {confidence:.2f}% confidence")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

with tab2:
    st.header("Live Webcam Recognition")
    st.write("Click 'Start' to begin live webcam detection")
    
    # Session state for webcam
    if 'stop_webcam' not in st.session_state:
        st.session_state['stop_webcam'] = False
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button('Start')
    with col2:
        stop_button = st.button('Stop')
    
    if stop_button:
        st.session_state['stop_webcam'] = True
        
    if start_button:
        st.session_state['stop_webcam'] = False
        
        # Create placeholder for webcam feed
        video_placeholder = st.empty()
        result_placeholder = st.empty()
        
        # Start webcam
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened() and not st.session_state['stop_webcam']:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam")
                break
                
            # Save current frame
            cv2.imwrite("temp_webcam.png", frame)
            
            try:
                # Process with multi-digit recognition
                results = predict_digit("temp_webcam.png", multi_digit=True)
                
                # Draw predictions
                for result in results:
                    digit = result['digit']
                    confidence = result['confidence']
                    pos = result['position']
                    
                    cv2.rectangle(frame, 
                                (pos['x'], pos['y']), 
                                (pos['x'] + pos['width'], pos['y'] + pos['height']), 
                                (0, 255, 0), 2)
                    
                    cv2.putText(frame, f"{digit} ({confidence:.1f}%)", 
                              (pos['x'], pos['y'] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              (0, 255, 0), 2)
                
                # Update video feed
                video_placeholder.image(frame, channels="BGR", use_column_width=True)
                
                if results:
                    result_text = " ".join([str(r['digit']) for r in results])
                    result_placeholder.write(f"Detected number: {result_text}")
                else:
                    result_placeholder.write("No digits detected")
                    
            except Exception as e:
                st.error(f"Error processing frame: {str(e)}")
                break
        
        # Release webcam when stopped
        cap.release()
        
    st.write("Note: Click 'Stop' to end the webcam feed")

with tab3:
    st.header("Model Performance")
    st.subheader("Confusion Matrix")
    
    if st.button("Generate Confusion Matrix"):
        # Get confusion matrix
        cm = generate_confusion_matrix()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Display the plot
        st.pyplot(fig)
        
        # Calculate and display accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        st.write(f"Overall Accuracy: {accuracy:.2%}")
