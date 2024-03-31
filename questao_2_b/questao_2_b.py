"""

@author: Karoline da Rocha
"""

"""
2. Seja o conjunto de imagens satelitais associadas às regiões urbanas (pasta Questão 2)
    b.    Considerando um conjunto de imagens satelitais provenientes do sensor WorldView-2 (resolução de 0.5m e 8 bandas espectrais [1],
          em arquivos .tif), implemente algum método para segmentação da vegetação presente nas cenas.
"""

# Bibliotecas:
import tifffile
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from skimage import exposure, filters
from sklearn.cluster import KMeans

# Caminho para as imagens TIFF
caminho_imagens = ['8band_AOI_1_RIO_img46.tif', '8band_AOI_1_RIO_img47.tif', '8band_AOI_1_RIO_img66.tif', '8band_AOI_1_RIO_img67.tif']

# Configuração da figura e subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

"""
 Explicação das 8 bandas espectrais:
     
    0. Banda Espectral Panchromática (Pan);

    1. Banda Blue;

    2. Banda Green: A banda verde é comumente usada na detecção e classificação de vegetação 
                            saudável, pois as plantas refletem fortemente nesta faixa espectral.

    3. Banda Yellow: A banda amarela pode ser útil para diferenciar entre diferentes tipos de 
                              vegetação, solo exposto e culturas agrícolas.

    4. Banda Red: A banda vermelha é útil na detecção de mudanças na cobertura vegetal, 
                            identificação de áreas urbanas, detecção de incêndios florestais e 
                            monitoramento de culturas agrícolas.

    5. Banda Red Edge: A banda de borda vermelha é especialmente útil na análise da 
                                         saúde das plantas e no mapeamento da cobertura vegetal. 
                                         É sensível a mudanças sutis na vegetação, como estresse hídrico 
                                         ou doenças.

    6. Banda Infravermelha Próxima 1 (Near Infrared 1);

    7. Banda Infravermelha Próxima 2 (Near Infrared 2);
"""

# Exibindo uma das 8 bandas das 4 imagens fornecidas:

"""
 Analisando o que cada banda representa, para a segmentação da vegetação as mais indicadas seriam:
     -  Banda Green (2);
     -  Banda Yellow (3);
     -  Banda Red (4);
     -  Banda Red Edge (5);
     
     
"""
for i, caminho in enumerate(caminho_imagens):
    # Abrir a imagem TIFF
    imagem = tifffile.imread(caminho)
    
    # Selecionando a primeira banda espectral para visualização.
    banda_1 = imagem[:,:,5]
    
    # Plota a imagem no subplot correspondente
    axs[i//2, i%2].imshow(banda_1, cmap='gray')
    axs[i//2, i%2].axis('off')

# Ilustras as imagens lidas.
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------
# Tentando realizar a segmentação da vegetação.

# Carregando as imagens:
imagens = []
for path in caminho_imagens:
    with rasterio.open(path) as src:
        img = src.read()
        imagens.append(img)


# Normalizando as imagens:
imagens_normalizadas = []
for img in imagens:
    img_normalized = exposure.equalize_hist(img)
    imagens_normalizadas.append(img_normalized)


# Extraindo as características das imagens usando "Index Ratio" NDVI(Normalized Difference Vegetation Index).
"""
    De acordo com [1], esse index identifica áreas de vegetação e determine a saúde de cada classe de vegetação.
    Dado qualquer outro sistema MSI(Multispectral imagery - Imagens multiespectrais), uma faixa vermelha é usada 
    para representar o baixo nível de refletância da vegetação e um NIR amplo para representar os valores de refletância 
    mais elevados. A banda vermelha permanece  fiel a níveis de refletância mais baixos do que a banda NIR2, que tem um 
    valor mais alto do que as bandas NIR largas tradicionais, portanto deve produzir um valor de NDVI mais alto.
    
    Formula do NDVI = (banda_red-banda_NRI2)/(banda_red+banda_NRI2)
    
    [1] - https://www.nv5geospatialsoftware.com/portals/0/pdfs/envi/8_bands_Antonio_Wolf.pdf
"""
ndvi_imagens = []
for img in imagens_normalizadas:
    red_band = img[4]
    nir2_band = img[7]
    ndvi = (red_band - nir2_band) / (red_band + nir2_band)
    ndvi_imagens.append(ndvi)


# Realizando a segmentação usando k-means.
segmentando_vegetacao_imagens = []
for ndvi in ndvi_imagens:
    kmeans = KMeans(n_clusters=3)
    ndvi_reshaped = ndvi.reshape((-1, 1))
    kmeans.fit(ndvi_reshaped)
    segmentando_vegetacao = kmeans.labels_.reshape(ndvi.shape)
    segmentando_vegetacao_imagens.append(segmentando_vegetacao)

# Removendo o ruído e visualizando a segmentação obtida.
plt.figure(figsize=(10, 10))
for i, segmentando_vegetacao in enumerate(segmentando_vegetacao_imagens):
    plt.subplot(2, 2, i+1)
    
    # Eliminando ruídos da imagem.
    segmentando_vegetacao = filters.median(segmentando_vegetacao, footprint=np.ones((3,3)))
    plt.imshow(segmentando_vegetacao, cmap='gray')
    plt.title(f'Imagem {i+1}')
    plt.axis('off')

plt.suptitle('Segmentação da Vegetação')
plt.show()

"""
    Comentário: A segmentação não funcionou corretamente.
"""

