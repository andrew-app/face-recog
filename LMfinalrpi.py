import cv2
import numpy as np
import dlib


class faceLM:
    
    
    def __init__(self,pic):
        #Define empty lists to store landmarks for x and y coordinates
        xpoints = [0] * 68
        ypoints = [0] * 68
        #import dlib tools
        detect = dlib.get_frontal_face_detector()
        predict = dlib.shape_predictor("/media/pi/FS/OpenCV/FaceLM/shape_predictor_68_face_landmarks.dat")
        #read image for analysis
        img = cv2.imread(pic)
        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #face detect function dlib
        faces = detect(gray)
       #landmark prediction
        for face in faces:
            landmark = predict(gray, face)
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
           
        
        
            


import os

path = '\\media\\pi\\FS\\OpenCV\\FaceLM\\images\\'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path): #Generate the file names in a directory tree by walking the tree either top-down or bottom-up
    for file in f: #read files that are in folder
        print(file)
        if '.jpg' in file: #only care about jpg
            files.append(os.path.join(r, file))
print(files)


for i in range(0,4):
    a = faceLM(files[i])

    b = faceLM.x(a)

    print(b)
        

