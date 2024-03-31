"""

@author: Karoline da Rocha
"""

import cv2
import numpy as np

"""
    Função que calcula a PSNR (Peak Signal-to-Noise Ratio, em português, "Razão Pico-Sinal-Ruído").
    O PSNR é uma medida de qualidade de imagem que compara o sinal original (a imagem sem ruído) com a imagem com ruído ou imagem
    filtrada. Valores mais altos de PSNR indicam que a imagem filtrada está mais próxima da imagem original.
"""
def psnr(original, filtrada):
    mse = np.mean((original - filtrada) ** 2)
    if mse == 0:
        return float('inf')  
    """  Verificando se o MSE calculado é zero. 
     Se for, isso significa que as duas imagens são idênticas (ou seja, não há erro), logo, o PSNR é definido como infinito."""
                            
        
    max_pixel = 255.0
    valor_psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) # Calculando o valor da PSNR.
    return valor_psnr

# Carregando as imagens fornecidas:
imagem_originalSemruido = cv2.imread('airport_gray.png', cv2.IMREAD_GRAYSCALE)
imagem_ruidosa = cv2.imread('airport_gray_noisy.png', cv2.IMREAD_GRAYSCALE)

# Verificando o valor PSNR antes de aplicar o filtro mediano na imagem:
psnr_antesProcess = psnr(imagem_originalSemruido, imagem_ruidosa)
print("PSNR antes do processamento:", round(psnr_antesProcess, 4), "dB")

# Aplicando o filtro mediano para remover o ruído ruído de fundo salt & pepper da imagem.
tam_kernel = 7 # tamanho do kernel, que determina a vizinhança sobre a qual a mediana é calculada. 
imagem_filtrada = cv2.medianBlur(imagem_ruidosa, tam_kernel) 

# Salvar a imagem filtrada
cv2.imwrite('airport_gray_filtrado.png', imagem_filtrada)

# Verificando o valor PSNR após aplicar o filtro mediano na imagem:
psnr_depoisProcess = psnr(imagem_originalSemruido, imagem_filtrada)
print("PSNR depois de aplicar o filtro mediano:", round(psnr_depoisProcess, 4), "dB")

# Mostrar a imagem original e a imagem sem ruído
cv2.imshow('Imagem Original Sem Ruido', imagem_originalSemruido)
cv2.imshow('Imagem Com Ruido', imagem_ruidosa)
cv2.imshow('Imagem Filtrada', imagem_filtrada)
cv2.waitKey(0)
cv2.destroyAllWindows()
