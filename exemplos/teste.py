import cv2 as cv
from os import listdir, path, makedirs
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt

faces_path_teste = "reconhecimento_facial/img/teste/"

def padronizar_imagens(img_caminho):
    imagem = cv.imread(img_caminho, cv.IMREAD_GRAYSCALE) #ja abre a imagem em escala de cinza
    imagem = cv.resize(imagem, (200,200), interpolation=cv.INTER_LANCZOS4) #imagem, tamanho de saida, e interpolação

    return imagem
lista_faces_teste = [f for f in listdir(faces_path_teste) if isfile(join(faces_path_teste, f))]

dados_teste, sujeitos_teste = [],[]

for i, arq in enumerate(lista_faces_teste):
    img_path = faces_path_teste + arq
    imagem = padronizar_imagens(img_path)
    dados_teste.append(imagem)
    sujeito = arq[1:3]
    sujeitos_teste.append(int(sujeito))

sujeitos_teste = np.asarray(sujeitos_teste, dtype=np.int32)

plt.figure(figsize=(20,10))

plt.subplot(121)
plt.title("Sujeito " + str(sujeitos_teste[6]))
plt.imshow(dados_teste[6], cmap="gray")

plt.subplot(122)
plt.title("Sujeito " + str(sujeitos_teste[7]))
plt.imshow(dados_teste[7], cmap="gray")

plt.show()