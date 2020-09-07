import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

#labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

train_path = "data/train+val"
test_path = "data/test"

#Training set augentation and normalization
train_datagen = ImageDataGenerator(rescale=1/255,
							rotation_range=30,
							shear_range=0.3,
							zoom_range=0.3,
							width_shift_range=0.4,
							height_shift_range=0.4,
							horizontal_flip=True,
							fill_mode='nearest')

#Testing set normalization
test_datagen = ImageDataGenerator(rescale=1/255)

#Get training data "train_path" directory
train = train_datagen.flow_from_directory(train_path,
										batch_size=32,
										target_size=(48,48),
										shuffle= True)

#Get testing data "test_path" directory
test = test_datagen.flow_from_directory(test_path,
										batch_size=32,
										target_size=(48,48),
										shuffle= True)


model = tf.keras.Sequential([Conv2D(96, (3,3), activation='relu', padding='same', input_shape= (48,48,1)),
                             BatchNormalization(),
                             Conv2D(96, (3,3), activation='relu', padding='same'),
                             BatchNormalization(),
                             MaxPooling2D(2,2),
                             
                             Conv2D(192, (3,3), activation='relu', padding='same'),
                             BatchNormalization(),
                             Conv2D(256, (3,3), activation='relu', padding='same'),
                             BatchNormalization(),
                             MaxPooling2D(2,2),
                             
                             Conv2D(256, (3,3), activation='relu', padding='same'),
                             BatchNormalization(),
                             Conv2D(256, (3,3), activation='relu', padding='same'),
                             BatchNormalization(),
                             MaxPooling2D(2,2),
                             
                             Conv2D(256, (3,3), activation='relu', padding='same'),
                             BatchNormalization(),
                             Conv2D(256, (3,3), activation='relu', padding='same'),
                             
                             Flatten(),
                             Dense(256, activation='relu'),
                             BatchNormalization(),
                             Dropout(0.25),
                             Dense(6, activation='softmax')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#The actually traning was done on Kaggle's servers with a GPU
model.fit(train ,epochs=150)

model.save('emotion_model_new.h5')
#This model achieves 71% test accuracy
model.evaluate(test, verbose=1)

