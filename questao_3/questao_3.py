"""

@author: Karoline da Rocha
"""

# Bibliotecas:
import json

# Caminho dos arquivos JSON.
caminho_arquivo_1 = 'building/annotation.json'
caminho_arquivo_2 = 'building/annotation-small.json'

# Abrindo o arquivo "annotation.json", em modo de leitura
with open(caminho_arquivo_1, 'r') as arquivo:
    
    # Carregando os dados lidos.
    dados_1 = json.load(arquivo)
    
# Abrindo o arquivo "annotation-small.json", em modo de leitura
with open(caminho_arquivo_2, 'r') as arquivo:
    
    # Carregando os dados lidos.
    dados_2 = json.load(arquivo)

info_img = dados_1['images'];

anotacao_img = dados_1['annotations'];

"""
    Comentario: Resolução incompleta.
"""
