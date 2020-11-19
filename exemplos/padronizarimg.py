import cv2 as cv
from os import listdir, path, makedirs
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

faces_path_treino = "reconhecimento_facial/img/treino/"
faces_path_teste = "reconhecimento_facial/img/teste/"

def padronizar_imagens(img_caminho):
    imagem = cv.imread(img_caminho, cv.IMREAD_GRAYSCALE) #ja abre a imagem em escala de cinza
    imagem = cv.resize(imagem, (200,200), interpolation=cv.INTER_LANCZOS4) #imagem, tamanho de saida, e interpolação

    return imagem

lista_faces_treino = [f for f in listdir(faces_path_treino) if isfile(join(faces_path_treino , f ))] #pega todas as imagens do diretorio de treino
lista_faces_teste = [f for f in listdir(faces_path_teste) if isfile(join(faces_path_teste, f))]

dados_treinamento, sujeitos = [],[]
dados_teste, sujeitos_teste = [],[]

for i, arq in enumerate(lista_faces_treino): #padroniza a imagem e add a lista de dados de treinamento
    img_path = faces_path_treino + arq
    imagem = padronizar_imagens(img_path)
    dados_treinamento.append(imagem)
    sujeito = arq[1:3]
    sujeitos.append(int(sujeito))

for i, arq in enumerate(lista_faces_teste):
    img_path = faces_path_teste + arq
    imagem = padronizar_imagens(img_path)
    dados_teste.append(imagem)
    sujeito = arq[1:3]
    sujeitos_teste.append(int(sujeito))

sujeitos = np.asarray(sujeitos, dtype=np.int32)
sujeitos_teste = np.asarray(sujeitos_teste, dtype=np.int32)
