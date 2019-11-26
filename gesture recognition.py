q# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 01:10:19 2019

@author: hp
"""
import cv2
import imutils
import numpy as np
from PIL import Image
from keras import models
from keras.models import load_model
predictions = []
model = models
model = model.load_model('model.h5')
bg = None
def run_avg (image, aweight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image,bg,aweight)
def segment ( image, threshold =25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts, hierarchy= cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key = cv2.contourArea)
        return ( thresholded,segmented)
if __name__ == "__main__":
    aweight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    while(1):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width = 700)
        frame = cv2.flip(frame,1)
        clone = frame.copy()
        (heght,width) = frame.shape[:2]
        roi = frame [ top:bottom, right:left]
        gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
       # gray = cv2.GaussianBlur(gray,(7,7),0)
        if num_frames < 60:
            run_avg (gray,aweight)
            num_frames += 1
            print(num_frames)
        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded,segmented) = hand
                cv2.drawContours(clone, [segmented + (right,top)], -1,(0,0,255))
                cv2.imshow("thresholded",thresholded)
            
                cv2.rectangle(clone,(left,top),(right,bottom),(0,255,0),2)
                cv2.imshow("video feed", clone)
                image = Image.fromarray(thresholded)
                im = image.resize((320,120))
                im = np.array(im)
                im = np.array(im , dtype = 'float32')
                im = im.reshape((1,120,320,1))
                im = im/255
                prediction = model.predict(im)
                prediction = prediction.reshape((10,1))
                prediction -= 1
                prediction *= prediction
                minimum = 0
                for j in range(9):
                    if(prediction[minimum]>prediction[j]):
                        minimum = j
                print(minimum)
                predictions.append(minimum)
                keypress = cv2.waitKey(1) &0xFF
                if keypress == ord("q"):
                    break
camera.release()
cv2.destroyAllWindows()
