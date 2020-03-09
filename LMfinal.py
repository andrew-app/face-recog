import cv2
import numpy as np
import dlib
import pylab

class faceLM:
    
    
    def __init__(self,pic):
        #Define empty lists to store lm for x and y coordinates
        xpoints = [0] * 68
        ypoints = [0] * 68
        #import dlib tools
        detect = dlib.get_frontal_face_detector()
        predict = dlib.shape_predictor("F:\OpenCV\FaceLM\shape_predictor_68_face_landmarks.dat")
        #read image for analysis
        img = cv2.imread(pic)
        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #face detect function dlib
        faces = detect(gray)
       #landmark prediction
        for face in faces:
            landmark = predict(gray,face)
        #get coordinates of each x and y in landmark data and store in respective list
        for i in range(0,68):
            x = landmark.part(i).x
            y = landmark.part(i).y
            xpoints[i] = x
            ypoints[i] = y
        self.x = xpoints
        self.y = ypoints
           
    def x(self):
        return self.x

    def y(self):
        return self.y
           
        
        
            

cap = "F:\OpenCV\FaceLM\keanu1.jpg"

a = faceLM(cap)

b = faceLM.x(a)

c = faceLM.y(a)
print(c)    
        

