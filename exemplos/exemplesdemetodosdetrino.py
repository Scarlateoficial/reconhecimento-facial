

"""
modelo_eingenfaces = cv.face.EigenFaceRecognizer_create()
modelo_eingenfaces.train(dados_treinamento, sujeitos)

predicao = modelo_eingenfaces.predict(dados_teste[6])
print(predicao)
predicao = modelo_eingenfaces.predict(dados_teste[7])
print(predicao)
"""
"""
modelo_fisherfaces = cv.face.FisherFaceRecognizer_create()
modelo_fisherfaces.train(dados_treinamento, sujeitos)

predicao = modelo_fisherfaces.predict(dados_teste[13])
print(predicao)
predicao = modelo_fisherfaces.predict(dados_teste[19])
print(predicao)
"""

"""
modelo_lbph = cv.face.LBPHFaceRecognizer_create()
modelo_lbph.train(dados_treinamento, sujeitos)

predicao = modelo_lbph.predict(dados_teste[21])
print(predicao)
predicao = modelo_lbph.predict(dados_teste[27])
print(predicao)

plt.figure(figsize=(20,10))

plt.subplot(121)
plt.title("Sujeito " + str(sujeitos_teste[21]))
plt.imshow(dados_teste[21], cmap="gray")

plt.subplot(122)

plt.title("Sujeito " + str(sujeitos_teste[27]))
plt.imshow(dados_teste[27], cmap="gray")

plt.show()
"""