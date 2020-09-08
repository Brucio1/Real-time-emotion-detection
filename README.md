# Real-time-emotion-detection
This program uses your local webcam to recognise 6 different emotions; Happiness, Sadness, Angry, Fear, Surprise and Neutral. Everything happens in real time using OpenCV and Convolution Neural Networks. 

# Installation
- Python 3.6 or below
- Tensorflow 2.0 or higher
- OpenCV
- Numpy

# Dataset
Kaggle dataset: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

The original dataset had 7 different class labels. I got rid of "Disgust" as it only had <500 training examples, not enough to properly train the model.
