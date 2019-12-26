import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()
classifier.add(Convolution2D(64,3,3,input_shape=(100,100,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(64,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Flatten())

classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('trainingImages', target_size=(100, 100),batch_size=32,class_mode='binary')

validation_generator = test_datagen.flow_from_directory('testingImages',target_size=(100, 100), batch_size=32,class_mode='binary')

classifier.fit_generator(train_generator,steps_per_epoch=745,epochs=25,validation_data=validation_generator,validation_steps=103)


