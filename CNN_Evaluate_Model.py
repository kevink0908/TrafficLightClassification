import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image

# Load the trained model.
model = load_model('traffic_light_model.h5')

# Create classes for the three different traffic light colors.
classes = {0: 'Red', 1: 'Green', 2: 'Yellow'}

# Test on an image.
def test_on_img(img_path):
    image = Image.open(img_path)
    image = image.resize((30, 30))
    data = np.array([np.array(image)])
    Y_pred = model.predict(data)
    prediction = np.argmax(Y_pred, axis=1)
    return image, classes[prediction[0]]

def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        plot, prediction = test_on_img(file_path)
        
        # Create a figure and axes.
        # NOTE: Adjust the figure size as needed here.
        fig, ax = plt.subplots(figsize=(8, 6))  
        
        # Display the image.
        ax.imshow(plot)
        
        # Add the prediction text to the plot.
        ax.text(0.5, -0.1, f"Predicted traffic light color: {prediction}", fontsize=12, color='black', ha='center', transform=ax.transAxes)
        
        # Display the plot.
        plt.show()

# Create a Tkinter GUI.
root = tk.Tk()
root.title("Traffic Light Classification")

# Prompt the user to choose an image to test.
label = tk.Label(root, text="Choose light image to test", font=("Arial", 14))
label.pack(pady=20)

# Perform traffic light classification once the user selects an image.
browse_button = tk.Button(root, text="Browse Image", command=browse_file)
browse_button.pack(pady=10)

root.mainloop()
