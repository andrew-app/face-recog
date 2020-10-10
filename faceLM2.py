import cv2
import numpy as np
import dlib

pic2 = cv2.imread("F:\OpenCV\FaceLM\hugh1.jpg")

detect = dlib.get_frontal_face_detector()

gray2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)

predict = dlib.shape_predictor("F:\OpenCV\FaceLM\shape_predictor_68_face_landmarks.dat")


faces = detect(gray)


# Getting each point from the coordinates of rectangle vertices can not use array indexing to get tuples with dlib
for face in faces:
    print(face)
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    landmarkF2 = predict(gray,face)

    for i in range(0,68):
        xs = landmarkF2.part(i).x
        ys = landmarkF2.part(i).y
        
        xpts[i] = x
        ypts[i] = y


cv2.rectangle(cap,(x1,y1),(x2,y2), (255,0,0), 2)


cv2.imshow('frame',cap)

cv2.waitKey()


cv2.destroyAllWindows()
