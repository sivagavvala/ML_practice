#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:21:08 2019

@author: siva
"""


# Buliding the CNN

# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#If Using Theano backend...input_shape = (3, 3, 64)

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
## Adding 2nd Convolution layer
#classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))


#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Part 2 - Fitting CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                                rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '/home/siva/Desktop/DS_ML/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset1/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '/home/siva/Desktop/DS_ML/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset1/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=500,
        epochs=15,
        validation_data=test_set,
        validation_steps=50)


#
#
#def getPrediction(imgStr):
#    test_image = image.load_img(imgStr, target_size = (64, 64))
#    test_image = image.img_to_array(test_image)/255.
#    test_image = np.expand_dims(test_image, axis = 0)
#    result = classifier.predict_classes(test_image)
#    training_set.class_indices # shows {'cats': 0, 'dogs': 1}
#    if result[0][0] == 1:
#        return 'dog'
#    else:
#        return 'cat'
#
#
#prediction = getPrediction('/home/siva/Desktop/DS_ML/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set/cats/cat.4001.jpg') 
#


