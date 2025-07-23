# Handwritten Digit Recognition Project

This project implements a deep learning model for recognizing handwritten digits using TensorFlow and OpenCV. The system supports multiple ways of digit recognition:
- Single digit recognition from uploaded images
- Multi-digit recognition from uploaded images
- Real-time digit recognition using webcam
- Interactive drawing and recognition

## Features

- **Model Architecture**: CNN-based model trained on MNIST dataset
- **Multiple Input Methods**:
  - Image upload
  - Webcam capture
  - Interactive drawing
- **Multiple Recognition Modes**:
  - Single digit recognition
  - Multi-digit recognition
  - Real-time recognition
- **Visual Feedback**:
  - Bounding boxes around detected digits
  - Confidence scores
  - Multiple prediction options

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Claudiuzah/AI-Model-Digit-Training.git
cd AI-Model-Digit-Training
```

2. Install the required dependencies:
```bash
pip install tensorflow opencv-python numpy streamlit pillow matplotlib seaborn scikit-learn
```

## Usage

### Training the Model
```bash
python training.py
```
This will train the CNN model on the MNIST dataset and save it as 'mnist_model.h5'.

### Running the Applications

1. Streamlit Web Interface:
```bash
streamlit run streamlit_app.py
```
Features:
- Upload images for digit recognition
- Use webcam for real-time recognition
- Toggle between single and multi-digit recognition
- View model performance metrics

2. Interactive Drawing Interface:
```bash
python draw_predict.py
```
Features:
- Draw digits using your mouse
- Get real-time predictions
- Clear canvas option

3. Standalone Webcam Application:
```bash
python webcam_predict.py
```
Features:
- Real-time digit recognition from webcam feed
- Multi-digit support
- Visual bounding boxes and confidence scores

## Project Structure

- `training.py`: CNN model training script using MNIST dataset
- `predict.py`: Core prediction functionality with single and multi-digit support
- `streamlit_app.py`: Web interface with multiple input methods
- `draw_predict.py`: Tkinter-based drawing interface
- `webcam_predict.py`: Standalone webcam recognition application
- `mnist_model.h5`: Pre-trained model file

## Model Architecture

The model uses a Convolutional Neural Network (CNN) with:
- 3 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- Dense layers for classification
- Softmax output for digit probabilities

## Performance and Limitations

- Training accuracy: ~99%
- Test accuracy: ~98%
- Best results with:
  - Clear handwriting
  - Good lighting conditions
  - High contrast between digits and background
  - Proper digit separation for multi-digit recognition

