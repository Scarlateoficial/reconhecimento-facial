import cv2 as cv
import numpy as np
from core import verificaruser, tratarimg

matricula = input("Digite sua matricula: \n")
email = input("Digite seu email: \n")
password = input("Digite sua senha: \n")

id = verificaruser.auth_user(matricula, email, password)

if id != False:

    classificador_face = cv.CascadeClassifier('classificadores/haarcascade_frontalface_default.xml')
    classificador_olho = cv.CascadeClassifier('classificadores/haarcascade_eye.xml')

    captura_video = cv.VideoCapture(0)

    try:

        contador = 0

        while(True):
            captura_ok, frame = captura_video.read()

            if captura_ok:
                frame = tratarimg.padronizar_imagem(frame)
                frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

                faces = classificador_face.detectMultiScale(frame_gray, 1.3, 5) #detecta as faces
                olhos = classificador_olho.detectMultiScale(frame_gray, 1.3, 5) #detecta os olhos

                if len(faces) > 0:                                              #se ele achou uma face
                    contador += 1                                               #face aumenta mais um
                    
                    if contador <= 100:
                        for (x,y,w,h) in faces:
                            roi = frame_gray[y:y+h, x:x+w]
                            cv.resize(roi, (200, 200), interpolation=cv.INTER_LANCZOS4)
                            cv.imwrite("dataset/" + "u"+ str(id) + "_" +str(contador) + ".png", roi)
                        
                        cv.putText(frame, "Coletado " + str(contador) + " faces", (20,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2) #mostra as faces já coletadas
                    
                    else: #quando acabar ele fecha
                        cv.putText(frame, "ConcluIdo", (20,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                        captura_video.release()
                        cv.destroyAllWindows()
                        print("Concluido")
                        break
                
                for (x,y,w,h) in faces: #faz um retangulo mostrando que encontrou o rosto
                    cv.rectangle(frame, (x,y), (x+w, y+h), (255,255,0),2) #imagem, extremidades do retangulo, cor, expessuras

                for (ex, ey, ew, eh) in olhos:
                    cv.rectangle(frame, (ex, ey), (ex+ew, ey + eh), (255,255,0),2)

                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR) #opcional, pra cor não ficar bugada
                cv.imshow('janela',frame)

                k = cv.waitKey(30) & 0xff #pega a tecla esc

                if k == 27: #caso esc for apertado para
                    break

    except KeyboardInterrupt: #caso de ctrl + c no terminal, tbm para
        captura_video.release()
        cv.destroyAllWindows()
        print("Interrompido")

else:
    print("Usuario não encontrado ou dados incorretos.")