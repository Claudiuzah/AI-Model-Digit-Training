import tkinter as tk
from tkinter import Canvas, Button, Label
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        
        # Canvas for drawing
        self.canvas_size = 280
        self.canvas = Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=2, pady=5, padx=5)
        
        # Create PIL image for drawing
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind mouse events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.predict_digit)
        
        # Drawing settings
        self.old_x = None
        self.old_y = None
        
        # Buttons
        self.clear_button = Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0, pady=5)
        
        # Prediction label
        self.prediction_label = Label(root, text="Draw a digit above", font=('Arial', 24))
        self.prediction_label.grid(row=2, column=0, columnspan=2)

    def paint(self, event):
        if self.old_x and self.old_y:
            # Draw on canvas
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                  width=20, fill='white', capstyle=tk.ROUND,
                                  smooth=tk.TRUE)
            # Draw on PIL image
            self.draw.line([self.old_x, self.old_y, event.x, event.y],
                         fill='white', width=20)
        
        self.old_x = event.x
        self.old_y = event.y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Draw a digit above")
        
    def preprocess_image(self):
        # Resize image to 28x28
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img_resized)
        img_array = img_array / 255.0
        
        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28, 1)
        return img_array

    def predict_digit(self, event):
        # Reset coordinates
        self.old_x = None
        self.old_y = None
        
        # Preprocess the image
        processed_image = self.preprocess_image()
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit] * 100
        
        # Update label
        self.prediction_label.config(
            text=f"Predicted: {predicted_digit} ({confidence:.1f}%)")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
