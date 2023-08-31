import cv2
import numpy as np

# Load the pre-trained Caffe models for age and gender
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

# Define the list of age ranges and gender labels
age_ranges = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
gender_labels = ["Male", "Female"]

# Load the image using OpenCV
image = cv2.imread("image.jpg")

# Define a function to predict age and gender
def predict_age_and_gender(image):
    # Resize the image to the input size expected by the models
    blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_ranges[np.argmax(age_preds)]

    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_labels[np.argmax(gender_preds)]

    return age, gender

# Get age and gender predictions
age, gender = predict_age_and_gender(image)

# Print the results
print(f"Age: {age}")
print(f"Gender: {gender}")
