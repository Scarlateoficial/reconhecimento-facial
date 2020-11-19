import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path = 'INRIAPerson/alura/imagens/px-people.jpg'

img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB) #converte a imagem para rgb
img_cinza = cv.cvtColor(img, cv.COLOR_RGB2GRAY) #converte a imagem para escala de cinza

classificador = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = classificador.detectMultiScale(img_cinza, 1.3, 5) #imagem, escala da imagem, ele vai diminuido a imagem pra pegar rosotos pequenos e grandes, ir testando, o 1.3 da 20 porcento, se diminuir pode achar um rosto

print(len(faces))

imagem_anotada = img.copy()

for (x,y,w,h) in faces:
    cv.rectangle(imagem_anotada, (x,y), (x+w, y+h), (255,255,0),2) #imagem, extremidades do retangulo, cor, expessura

imagem_da_vez = 0

for (x,y,w,h) in faces:
    imagem_da_vez += 1
    img_roi = img[y:y+h, x:x+w] #linha da altura, linha da largura
    img_roi = cv.cvtColor(img_roi, cv.COLOR_RGB2BGR)
    cv.imwrite("rostos/face_"+str(imagem_da_vez)+".png", img_roi) #salva a imagem

"""
img_roi = img[100:200, 1000:1200] #recorta uma parte da imagem
img_roi = cv.cvtColor(img_roi, cv.COLOR_BGR2RGB) #converte a imagem para rgb
cv.imwrite("img_roi.png", img_roi)
"""
plt.figure(figsize = (20,10))
plt.imshow(imagem_anotada)
plt.show()