"""

@author: Karoline da Rocha
"""

"""
2. Seja o conjunto de imagens satelitais associadas às regiões urbanas (pasta Questão 2)
    a.    A imagem RGB airport.png apresenta uma pista de aeroporto, conforme indicado abaixo. Implemente um código responsável 
          pela seleção dos conjuntos de pixels associados ao objeto silo.
"""

# Bibliotecas:
import cv2
import numpy as np

# Carregando a imagem
image = cv2.imread('airport.png')

"""
 Converter a imagem para o espaço de cores HSV (Hue, Saturation, Value).
 Como observei na imagem que a pista possui uma cor mais cinza que os demais objetos da imagem, optou-se por converter
 a imagem para o espaço de cores HSV. Esse espaço de cores é frequentemente usada em tarefas de processamento de imagem onde 
 a cor desempenha um papel importante como na detecção e segmentação de objetos baseada em cor.
"""

hsv_imagem = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir os intervalos de cor que acredita-se que destaque a pista do aeroporto.
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([50, 60, 255])  # Valores ajustados conforme necessário.
                                       # 50 é o valor máximo para o Hue, 60 é o valor máximo para a Saturação, e 255.

# Criando uma máscara para destacar apenas os pixels da pista do aeroporto.
mascara_1 = cv2.inRange(hsv_imagem, lower_bound, upper_bound)

# Aplicando a máscara criada na imagem original.
aero_pixels = cv2.bitwise_and(image, image, mask=mascara_1)

# Convertendo a imagem resultante para escala de cinza.
aero_cinza = cv2.cvtColor(aero_pixels, cv2.COLOR_BGR2GRAY)

"""
 Transformando a imagem resultante em uma imagem binária, em que os pixels são classificados como preto ou branco com base 
 no valor de intensidade do pixel em relação ao limiar de 110.
"""
thresh = cv2.threshold(aero_cinza,  110, 255, cv2.THRESH_BINARY)[1]

kernel = np.ones((3,3),np.uint8)

"""
 A operação de dilatação expande áreas brancas (pixels de valor 255) na imagem binária. 
 Isso é útil para preencher pequenos buracos e lacunas e para unir regiões próximas. 
"""
img_dilatacao = cv2.dilate(thresh,kernel,iterations =1)

"""
 A operação de erosão reduz áreas brancas na imagem binária. Ela é usada para remover ruídos pequenos, 
 eliminar pixels isolados.
"""
img_erosao = cv2.erode(img_dilatacao,kernel,iterations = 3)

# A dilatação e a erosão ajuda a remover ruídos e aprimorar a forma dos objetos na imagem binária.

# Utiliza-se o algoritmo Canny para detecção de bordas na imagem. 
bordas = cv2.Canny(img_erosao, 50,150)

"""
 Encontra os contornos na imagem que as bordas foram detectadas.
 cv2.RETR_EXTERNAL: Este parâmetro especifica o modo de recuperação de contornos. Significa que apenas os contornos externos são retornados. 
 cv2.CHAIN_APPROX_SIMPLE: Este parâmetro especifica o método de aproximação dos contornos, nesse método de compressão de 
 contorno que remove todos os pontos intermediários e armazena apenas os pontos extremos que definem o contorno. 
"""
contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Identica o maior contorno, que provavelmente representa a pista do aeroporto.
maior_contorno = max(contornos, key=cv2.contourArea)

# Destaca na imagem o maior contorno detectado.
imagem_comRetangulos = image.copy()  # Criar uma cópia da imagem original

x, y, w, h = cv2.boundingRect(maior_contorno)
cv2.rectangle(imagem_comRetangulos, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Desenha um retângulo em verde.

# Ilustra a imagem com o retângulo detectado.
cv2.imshow('Pista do aeroporto destacada', imagem_comRetangulos)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Cria uma máscara preenchida com zeros do mesmo tamanho da imagem original
mascara_2 = np.zeros_like(aero_cinza)

# Desenha o maior contorno na máscara
cv2.drawContours(mascara_2, [maior_contorno], -1, (255), thickness=cv2.FILLED)

# Aplica a máscara na imagem original
pixels_aero_destacados = cv2.bitwise_and(image, image, mask=mascara_2)

# Mostra a imagem com apenas os pixels do maior contorno detectado
cv2.imshow('Pixels da pista do aeroporto', pixels_aero_destacados)
cv2.waitKey(0)
cv2.destroyAllWindows()
