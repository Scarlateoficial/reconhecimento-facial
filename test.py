import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from os import listdir, path, makedirs
from os.path import isfile, join

modelo_lbph = cv.face.LBPHFaceRecognizer_create()
modelo_lbph.read("classificadores/lbph_trainigdata.xml")

faces_path_treino = "dataset/"

def padronizar_imagens(img_caminho):
    imagem = cv.imread(img_caminho, cv.IMREAD_GRAYSCALE) #ja abre a imagem em escala de cinza
    imagem = cv.resize(imagem, (200,200), interpolation=cv.INTER_LANCZOS4) #imagem, tamanho de saida, e interpolação

    return imagem

lista_faces_treino = [f for f in listdir(faces_path_treino) if isfile(join(faces_path_treino , f ))] #pega todas as imagens do diretorio de treino

dados_treinamento, sujeitos = [],[]

for i, arq in enumerate(lista_faces_treino): #padroniza a imagem e add a lista de dados de treinamento
    img_path = faces_path_treino + arq
    imagem = padronizar_imagens(img_path)
    dados_treinamento.append(imagem)
    sujeito = arq[1:2]
    sujeitos.append(int(sujeito))

sujeitos = np.asarray(sujeitos, dtype=np.int32)

predicao = modelo_lbph.predict(dados_treinamento[21])
print(predicao)
predicao = modelo_lbph.predict(dados_treinamento[27])
print(predicao)

plt.figure(figsize=(20,10))

plt.subplot(121)
plt.title("Sujeito " + str(sujeitos[21]))
plt.imshow(dados_treinamento[21], cmap="gray")

plt.subplot(122)

plt.title("Sujeito " + str(sujeitos[27]))
plt.imshow(dados_treinamento[27], cmap="gray")

plt.show()