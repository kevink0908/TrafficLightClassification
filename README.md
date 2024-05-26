# TrafficLightClassification

This repository contains the source code for a traffic classification program that uses Convolutional Neural Networks (CNN) to classify random images of traffic lights.

### How to Run:

1. Ensure that the "train" folder that contains all the images of traffic light images is downloaded in the same folder as "CNN_Evaluate_Model.py" and "CNN_Model_Training.py".
2. Run the "CNN_Model_Training.py" to begin training the model using an IDE of your choice.
3. After completing a certain number of training epochs, check to see if "traffic_light_model.h5" has been saved.
4. After "traffic_light_model.h5" has been saved, run the "CNN_Evaluate_Model.py" using IDE of your choice.
5. Once the GUI is displayed, click on the "Browse Image" button to select an image and begin traffic light classification.

Data Source: https://www.kaggle.com/datasets/sachsene/carla-traffic-lights-images and https://www.kaggle.com/datasets/chandanakuntala/cropped-lisa-traffic-light-dataset
