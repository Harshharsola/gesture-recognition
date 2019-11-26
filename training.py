# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:19:26 2019

@author: hp
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('c:/users/hp/desktop/machine learning/leapGestRecog/00/'):
    if not j.startswith('.'):
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1
lookup
 
x_data = []
y_data = []
datacount = 0
for i in range (0,10):
    for j in os.listdir('c:/users/hp/desktop/machine learning/leapGestRecog/0'+str(i)):
        if not j.startswith('.'):
            count = 0
            for k in os.listdir('c:/users/hp/desktop/machine learning/leapGestRecog/0'+str(i)+'/'+j+'/'):
                img = Image.open('c:/users/hp/desktop/machine learning/leapGestRecog/0'+str(i) + '/' +j +'/' + k).convert('L')
                img = img.resize((320,120))
                arr = np.array(img)
                x_data.append(arr)
                count = count + 1
            y_values = np.full((count,1),lookup[j])
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount,1)

from random import randint
for i in range (0,10):
    plt.imshow(x_data[i*200,:,:])
    plt.title(reverselookup[y_data[i*200,0]])
    plt.show()
    
import keras
from keras import layers
from keras.utils import to_categorical
y_data = to_categorical(y_data)
x_data = x_data.reshape((datacount,120,320,1))
x_data /= 255
from sklearn.model_selection import train_test_split
x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2) 
x_test,x_validation,y_test,y_validation = train_test_split(x_further,y_further,test_size = 0.5)
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32,(5,5), strides = (2,2), activation = 'relu', input_shape= (120,320,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=8, batch_size=64, verbose=1, validation_data=(x_validation, y_validation))
[loss,acc]=model.evaluate(x_test, y_test, verbose = 1)
print("accuracy" + str(acc))
model.save('model.h5')
model =models

model = model.load_model('model.h5')
import cv2
predictions = []
video =  cv2.VideoCapture(0)
while True:
    _,frame = video.read()
    image = Image.fromarray(frame, 'RGB')
    cv2.imshow("video feed", frame)
    im = image.resize((320,120))
    im = np.array(im)
    im = np.array(im , dtype = 'float32')
    im = im.reshape((3,120,320,1))
    im = im/255
    prediction = model.predict(im)
    predictions.append(prediction)
    keypress = cv2.waitKey(1) &0xFF
    if keypress == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
prediction = model.predict(x_test)
from sklearn.metrics import confusion_metrics
matrix = metrics.confusion_matrix(y_test.argmax(axis=1), prediction.argmax(axis=1))
predictions = prediction.argmax(axis=1)
y_tests = y_test.argmax(axis=1)
array = []
predictions = np.array(predictions, dtype = 'float32')
y_tests = np.array(y_tests, dtype = 'float32')
i=0
for i in predictions:
    if (predictions[i] == y_tests[i]):
        array.append(1)
    else :
        array.append(0)
        
image =  x_data[1]
image = Image.open('c:/users/hp/desktop/machine learning/leapGestRecog/00/01_palm/frame_00_01_0001.png').convert('L')
image = image.resize((320,120))
image = np.array(image)
image = np.array(image , dtype = 'float32')
image = image.reshape((1,120,320,1))
image = image/255
prediction = model.predict(image)
prediction
for i in prediction:
    if(prediction[i]==1):
        print(i)



