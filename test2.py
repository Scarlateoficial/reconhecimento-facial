import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from os import listdir, path, makedirs
from os.path import isfile, join

modelo_lbph = cv.face.LBPHFaceRecognizer_create()
modelo_lbph.read("classificadores/lbph_trainigdata.xml")
classificador_face = cv.CascadeClassifier('classificadores/haarcascade_frontalface_default.xml')

path = 'download(1).jpeg'

img = cv.imread(path, cv.IMREAD_GRAYSCALE)

faces = classificador_face.detectMultiScale(img, 1.3, 5) #detecta as faces

for (x,y,w,h) in faces:
    roi = img[y:y+h, x:x+w]
    cv.resize(roi, (200, 200), interpolation=cv.INTER_LANCZOS4)
    cv.imshow("roi",roi)

predicao = modelo_lbph.predict(roi)
print(predicao)