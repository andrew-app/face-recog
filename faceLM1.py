import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
import pylab
from img_rz import img_resize


pic1 = cv2.imread("F:\OpenCV\FaceLM\keanu1.jpg")
pic2 = img_resize("F:\OpenCV\FaceLM\keanu2.jpg")

detect = dlib.get_frontal_face_detector()

gray = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)

predict = dlib.shape_predictor("F:\OpenCV\FaceLM\shape_predictor_68_face_landmarks.dat")

faces1 = detect(gray)
faces2 = detect(gray2)

xpts = [0] * 68

ypts = [0] * 68

xref = [0] * 68

x2pts = [0] * 68

y2pts = [0] * 68

# Getting each point from the coordinates of rectangle vertices, cannot use array indexing to get tuples with dlib
#landmarking for pic1

for face in faces1:
    print(face)
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    landmarkF1 = predict(gray,face)

    for i in range(0,68):
        x = landmarkF1.part(i).x
        y = landmarkF1.part(i).y
        cv2.circle(pic1, (x,y), 2, (0,255,0), -1)
        xpts[i] = x
        ypts[i] = y
        xref[0] = 0
        if i > 0:

            xref[i] = i

cv2.rectangle(pic1,(x1,y1),(x2,y2), (255,0,0), 2)

cv2.imshow('frame',pic1)

#Landmarking for pic 2
for face in faces2:
    print(face)
    x3 = face.left()
    y3 = face.top()
    x4 = face.right()
    y4 = face.bottom()

    landmarkF2 = predict(gray2,face)

    for j in range(0,68):
        xx = landmarkF2.part(j).x
        yy = landmarkF2.part(j).y
        cv2.circle(pic2, (xx,yy), 2, (0,255,0), -1)
        x2pts[j] = xx
        y2pts[j] = yy
        xref[0] = 0
        if j > 0:

            xref[j] = j

cv2.rectangle(pic2,(x3,y3),(x4,y4), (255,0,0), 2)

cv2.imshow('frame2',pic2)        




fig, ax = plt.subplots()
xscat = np.asarray(xpts, dtype=np.float32)
y = np.asarray(ypts, dtype=np.float32)
hx = np.asarray(x2pts, dtype=np.float32)
hy = np.asarray(y2pts, dtype=np.float32)
x = np.asarray(xref, dtype=np.float32)

pylab.plot(x,y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
pylab.plot(x,p(x),'-r')
# the line equation:
#print "y=%.6fx+(%.6f)”%(z[0],z[1])


scatter = ax.scatter(x,xscat)
scatter2 = ax.scatter(x,y)
plt.show()


cv2.rectangle(pic1,(x1,y1),(x2,y2), (255,0,0), 2)

cv2.imshow('frame',pic1)
cv2.waitKey()
cv2.destroyAllWindows()

