import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Store data and labels in the list.
data = []
labels = []

# There will be three classes for three different traffic light colors.
# NOTE: 0 will be used for Red, 1 for Green, and 2 for Yellow.
classes = 3
cur_path = os.getcwd()

# Preprocess the images and adjust path generation for training images.
for i, color in enumerate(['red', 'green', 'yellow']):
    # Create the path to the color subdirectory.
    path = os.path.join(cur_path, 'train', color) 
    images = os.listdir(path)
    for a in images:
        try:
            # Use os.path.join for robust path construction.
            image = Image.open(os.path.join(path, a)) 
            image = image.resize((30,30))
            image = np.array(image)
            # Ensure all images have the same shape
            if image.shape == (30, 30, 3):  
                data.append(image)
                # The index i will be used as the label.
                labels.append(i)
            else:
                print(f"Ignoring image '{a}' in directory '{color}' due to shape mismatch.")
        except Exception as e:
            print(e)

# Convert lists into numpy arrays.
data = np.array(data)
labels = np.array(labels)

# Ensure the directory exists before saving data.
save_dir = './training/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save data and labels for future use.
np.save(os.path.join(save_dir, 'data'), data)
np.save(os.path.join(save_dir, 'target'), labels)

# Load data and labels.
data = np.load('./training/data.npy')
labels = np.load('./training/target.npy')
print(data.shape, labels.shape)

# Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Convert labels to one-hot encoding.
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# Build the CNN model.
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(classes, activation='softmax'))

# Compile the model.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

# Save the model in the Hierarchical Data Format (HDF).
model.save("traffic_light_model.h5")


