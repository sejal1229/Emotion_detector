# Emotion_detector
Simple Emotion detection using machine learning


# Real-Time Emotion Detection Using Deep Learning
This project demonstrates real-time emotion detection from facial expressions using a pre-trained Convolutional Neural Network (CNN) model. The system uses OpenCV for face detection and TensorFlow/Keras for emotion classification. A Streamlit web app is built to allow users to upload images and get instant emotion predictions.

Features
Real-time Emotion Detection: Accurately predicts emotions like Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
Face Detection: Detects human faces using OpenCV's Haar Cascade classifier.
User-Friendly Interface: Built with Streamlit to provide an intuitive and simple interface for users to upload images and see predictions.
Setup and Installation

How It Works
Model Architecture and Loading
The pre-trained CNN model is loaded from a saved .h5 file, which contains the model's architecture and learned weights.

python
Copy code
model = load_model('full_model.h5')

This model has been trained on a large dataset of facial expressions, and it classifies images into one of seven emotions.

Face Detection with Haar Cascade
OpenCV's Haar Cascade is used to detect faces within the uploaded image.

python
Copy code
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = facec.detectMultiScale(gray_image, 1.3, 5)

The classifier scans the image and returns the coordinates of any faces detected. If no face is detected, the system will return "No face detected."

Image Preprocessing
The detected face is cropped, resized to 48x48 pixels (the model's input size), and converted to grayscale before being passed to the CNN model.

python
Copy code
roi = cv2.resize(fc, (48, 48))
roi = roi[np.newaxis, :, :, np.newaxis]  # Adjust dimensions

Emotion Prediction
The preprocessed face is fed into the CNN model, which outputs a probability distribution across the seven possible emotions. The highest probability is selected as the predicted emotion.

python
Copy code
pred = EMOTIONS_LIST[np.argmax(model.predict(roi))]
Display Results
The result is displayed on the web app using Streamlit, along with the uploaded image.

python
Copy code
st.image(image, caption='Uploaded Image.', use_column_width=True)
st.write(f"Predicted Emotion: {emotion}")
Key Takeaways
Deep Learning Integration: This project combines traditional computer vision techniques with deep learning models for accurate emotion detection.
Interactive Web Interface: The Streamlit framework provides an easy-to-use interface for non-technical users to test the model.
Expandable: The model can be adapted for real-time video emotion detection or extended to more complex emotions and facial features.
Future Enhancements
Real-time Video Processing: Extend the model to handle live video streams for real-time emotion detection.
Advanced Emotions: Train the model to recognize more nuanced or mixed emotions.
Mobile Integration: Build a mobile-friendly version using frameworks like Flask or FastAPI for backend support.
