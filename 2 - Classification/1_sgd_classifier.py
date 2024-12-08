#==============================================================================
# Tópico 01 - Treinando um Classificador Binário com a função SGDClassifier
#==============================================================================

#------------------------------------------------------------------------------
# Importar a biblioteca Pandas e PyPlot
#------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Importar o conjunto de dados MNIST
#------------------------------------------------------------------------------

from sklearn.datasets import fetch_openml

#------------------------------------------------------------------------------
# Obtenção e impressão das chaves do conjunto de dados na tela
#------------------------------------------------------------------------------

conjunto_dados_mnist = fetch_openml('mnist_784', version=1)
print(f"\nChaves do Conjunto de Dados MNIST:\n {conjunto_dados_mnist.keys()}\n")
print("Uma chave 'DESC' descreve o conjunto de dados.")
print("Uma chave 'data' contém um array com uma linha por instância e uma coluna por característica.")
print("Uma chave 'target' contém um array com os rótulos.\n")

#------------------------------------------------------------------------------
# Separar os atributos e o alvo, exibindo suas dimensões
#------------------------------------------------------------------------------

atributos, rotulos = conjunto_dados_mnist['data'], conjunto_dados_mnist['target']
print(f"Dimensão das feactures: {atributos.shape}")
print(f"Dimensão os rótulos: {rotulos.shape}")
print(f"Feactures: \n{atributos}")
print(f"\nRotulos: \n{rotulos}")