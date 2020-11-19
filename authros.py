import cv2 as cv
import numpy as np
from core import verificaruser, tratarimg

captura_video = cv.VideoCapture(0)

classificador_face = cv.CascadeClassifier('classificadores/haarcascade_frontalface_default.xml')

contador = 0

model_lbph = cv.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 15)
model_lbph.read("classificadores/lbph_trainigdata.xml")

try:
    while(True):
        captura_ok, frame = captura_video.read()

        if captura_ok:
            frame = tratarimg.padronizar_imagem(frame)
            frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            faces = classificador_face.detectMultiScale(frame_gray, 1.3, 5)

            if len(faces) > 0:
                for (x,y,w,h) in faces:
                    roi = frame_gray[y:y+h, x:x+w]
                    roi = cv.resize(roi, (200, 200), interpolation=cv.INTER_LANCZOS4)
                    predicao = model_lbph.predict(roi)
                    cv.putText(frame, "Similaridade " + str(predicao[1]), (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                    if predicao[1] < 4.0:
                        r = verificaruser.coleta_dados(int(predicao[0]))
                        print(r)
                        break

            for (x,y,w,h) in faces: #faz um retangulo mostrando que encontrou o rosto
                cv.rectangle(frame, (x,y), (x+w, y+h), (255,255,0),2) #imagem, extremidades do retangulo, cor, expessuras

            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR) #opcional, pra cor não ficar bugada

            cv.imshow('janela',frame)

            k = cv.waitKey(30) & 0xff #pega a tecla esc
            if k == 27: #caso esc for apertado para
                break

except KeyboardInterrupt:
    captura_video.release()
    print("Interrompido")
