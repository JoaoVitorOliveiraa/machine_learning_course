#==============================================================================
# Tópico 01 - Treinando um Classificador Binário com a função SGDClassifier
#==============================================================================

#------------------------------------------------------------------------------
# Importar a biblioteca Pandas, PyPlot e Path
#------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#------------------------------------------------------------------------------
# Importar o conjunto de dados Digits
#------------------------------------------------------------------------------

caminho_dados_digits = Path('../data') / 'Digits.xlsx'
dados_digits = pd.read_excel(caminho_dados_digits)

dados_digits = dados_digits.iloc[:, 1:]   # Retirando a coluna dos id's.

atributos = dados_digits.iloc[:, :-1].values
rotulos = dados_digits.iloc[:, -1].values

#------------------------------------------------------------------------------
# Separar os atributos e o alvo, exibindo suas dimensões
#------------------------------------------------------------------------------

print(f"\nDimensão das features: {atributos.shape}")
print(f"\nDimensão dos rótulos: {rotulos.shape}")
print(f"\nFeatures: \n{atributos}")
print(f"\nRótulos: \n{rotulos}")

# ------------------------------------------------------------------------------
#  Visualizar um digito
# ------------------------------------------------------------------------------

# Cada imagem tem 8 x 8 pixels e cada característica representa a intensidade do pixel, do 0 (branco) a 16 (preto).

# Um imagem por digito.
# for index_digito in range(10):
#     digito = atributos[index_digito]
#     imagem_digito = digito.reshape(8, 8)
#     plt.imshow(imagem_digito, cmap='binary')
#     plt.show()

# Uma figura com a imagem de todos os dígitos.
plt.figure(figsize=(70, 50))
for i in range(0, 10):
    imagem_digito = plt.subplot(1, 10, i+1)
    imagem_digito.set_title("Rótulo = %.0f" % rotulos[i])

    imagem_digito.imshow(atributos[i, :].reshape(8, 8),
                  # interpolation='spline16',
                  # interpolation='nearest',
                  interpolation='none',
                  cmap='binary',
                  vmin=0, vmax=16)
    # plt.text(-8, 3, "y = %.2f" % y[i])

    imagem_digito.set_xticks(())
    imagem_digito.set_yticks(())

plt.show()

