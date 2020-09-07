import cv2
from tensorflow.keras.models import load_model
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = load_model('emotion_model_new.h5')

def get_prediction(face_image):
	#Resize and reshape to a (1, 48, 48, 1) for Conv2D on the model
	image = cv2.resize(face_image, (48,48)).reshape(1,48,48,1)
	image = image/255

	prediction = model.predict(image)
	#Double max() because predict's output is a 2D list
	probability = max(max(prediction))
	probability = "{:.2f}".format(probability*100)+'%'
	#Final prediction of the image
	prediction = np.argmax(prediction)

	return prediction, probability

def main():
		labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
		is_using_program = True
		camera = cv2.VideoCapture(0)

		while(is_using_program):
			#Get a single frame from the camera
			ret, frame = camera.read()
			#Convert the image into gray scale
			gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#Detect faces in the image
			faces = face_classifier.detectMultiScale(gray_image)

			for x, y, w, h in faces:
				#Draw a rectangle around the detected faces
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
				face_image = gray_image[y:y+h, x:x+w]
				prediction , probability = get_prediction(face_image)
				prediction = labels[prediction]+' '+probability

				cv2.putText(frame, prediction, (x,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), thickness=2)
			
			#Show the image with detected faces
			cv2.imshow("Emotion Detector", frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				camera.release()
				is_using_program = False
main()

cv2.destroyAllWindows()

