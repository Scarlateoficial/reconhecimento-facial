import cv2 as cv
import numpy as np

def padronizar_imagem(img):
    imagem = cv.resize(img, (400,300), interpolation=cv.INTER_LANCZOS4) #imagem, tamanho de saida, e interpolação
    imagem = cv.cvtColor(imagem, cv.COLOR_BGR2RGB)
    return imagem
