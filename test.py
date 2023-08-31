import cv2
import dlib
import tensorflow as tf
import numpy as np

# Load the pre-trained face detection model from dlib
detector = dlib.get_frontal_face_detector()

# Load the pre-trained facial landmarks model from dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the pre-trained age and gender models
age_model = tf.keras.models.load_model("age_model.h5")
gender_model = tf.keras.models.load_model("gender_model.h5")

def detect_face(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # Process each detected face
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)

        # Extract features from landmarks
        features = []
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            features.append(x)
            features.append(y)

        # Convert the features to a NumPy array
        features = np.array(features)

        # Resize the image for the age and gender models
        face_img = cv2.resize(image[face.top():face.bottom(), face.left():face.right()], (224, 224))
        face_img = np.expand_dims(face_img, axis=0)

        # Predict age and gender
        predicted_age = age_model.predict(face_img)[0]
        predicted_gender = gender_model.predict(face_img)[0]

        # Define the gender labels
        gender_labels = ['Male', 'Female']

        # Print the results
        print(f"Age: {int(predicted_age)} years")
        print(f"Gender: {gender_labels[np.argmax(predicted_gender)]}")

if __name__ == "__main__":
    image_path = "image.jpg"
    detect_face(image_path)
