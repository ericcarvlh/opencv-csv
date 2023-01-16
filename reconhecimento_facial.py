import cv2 as cv

# detecção de objetos
carregaAlgoritmo = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# 'lendo' a imagem
imagem = cv.imread('Fotos/imagem2t'
                   '.jpg')

# transformando a imagem em cinza para uma melhor leitura
imagemCinza = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)

# detectando as faces mesmo que haja tamanhos diferentes
# scaleFactor -> diminui a escala
faces = carregaAlgoritmo.detectMultiScale(imagemCinza, scaleFactor=1.08, minNeighbors=3)

# eixo x, eixo y, largura e alatura
for (x, y, l, a) in faces:
    # desenhando um retangulo em toda imagem que encontramos.
    cv.rectangle(imagem, (x, y), (x + l, y + a), (255, 255, 0), 2)
    # colocando um texto
    cv.putText(imagem, "unknown", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

# setando o que vou mostrando
cv.imshow('Faces', imagem)
# colocando um tempo de delay para abrir
cv.waitKey()
