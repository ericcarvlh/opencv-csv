import cv2

# scalefactor -> pega aquela imagem que esta muito grande e diminui ela.
# minNeighbors -> conforme aumenta o minNeighbors, maior a qualidade da imagem.
# minsize -> tamanho da face que ele tem que detectar.
carregaAlgoritmo = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

imagem = cv2.imread('fotos/imagem2.jpg')

# transformando a imagem para cinza porque a leitura é melhor.
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

faces = carregaAlgoritmo.detectMultiScale(imagemCinza)

# identificando as faces
print(faces)

for (x, y, l, a) in faces:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 255, 0), 2)

cv2.imshow("Faces", imagem)
cv2.waitKey()

"""
eixo X, eixoY, largura e altura
[150  98  69  69]
"""
