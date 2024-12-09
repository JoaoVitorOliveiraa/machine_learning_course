#==============================================================================
# Tópico 01 - Treinando um Classificador Binário com a função SGDClassifier
#==============================================================================

#------------------------------------------------------------------------------
# Importar a biblioteca Pandas, PyPlot e Path
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#------------------------------------------------------------------------------
# Importar o conjunto de dados Digits
#------------------------------------------------------------------------------

caminho_dados_digits = Path('../data') / 'Digits.xlsx'
dados_digits = pd.read_excel(caminho_dados_digits)
dados_digits = dados_digits.iloc[:, 1:]           # Retirando a coluna dos id's.

# ------------------------------------------------------------------------------
#  Exibição das primeiras 10 linhas do conjunto de dados através da função head()
# ------------------------------------------------------------------------------

print(f"\n\t\t-----Dez primeiras linhas do dataset-----\n")
print(dados_digits.head(n=10))

# ------------------------------------------------------------------------------
#  Descrição dos dados (como número de linhas, tipo de cada atributo e número de
#  valores não nulos) através da função info()
# ------------------------------------------------------------------------------

print("\n\n\t-----Descrição dos dados-----\n")
dados_digits.info()

# ------------------------------------------------------------------------------
#  Descobrindo quais categorias existem e quantos dígitos pertencem a cada
#  categoria usando a função value_counts()
# ------------------------------------------------------------------------------

print("\n\n\t-----Categorias do alvo, com suas respectivas quantidades-----\n")
print(dados_digits["target"].value_counts())

# ------------------------------------------------------------------------------
#  Resumo dos atributos numéricos através da função describe()
# ------------------------------------------------------------------------------

print(f"\n\n\t-----Resumo dos atributos numéricos-----\n")
print(dados_digits.describe())

#------------------------------------------------------------------------------
# Separar os atributos e o alvo, exibindo suas dimensões
#------------------------------------------------------------------------------

atributos = dados_digits.iloc[:, :-1].values
rotulos = dados_digits.iloc[:, -1].values
print("\n\n\t-----Dimensões-----")
print(f"\nDimensão das features: {atributos.shape}")
print(f"Dimensão dos rótulos: {rotulos.shape}\n")

# ------------------------------------------------------------------------------
#  Visualizar alguns digitos
# ------------------------------------------------------------------------------

# Cada imagem tem 8 x 8 pixels e cada característica representa a intensidade do
# pixel, do 0 (branco) a 16 (preto).

# Um imagem por digito (totalizando 10 imagens).
# for index_digito in range(10):
#     digito = atributos[index_digito]
#     imagem_digito = digito.reshape(8, 8)
#     plt.imshow(imagem_digito, cmap='binary')
#     plt.show()

# Uma figura armazenando as imagens dos 10 dígitos.
plt.figure(figsize=(70, 50))
for amostra in range(0, 10):
    imagem_digito = plt.subplot(1, 10, amostra+1)
    imagem_digito.set_title("Rótulo = %.0f" % rotulos[amostra])

    imagem_digito.imshow(atributos[amostra, :].reshape(8, 8),
                  # interpolation='spline16',
                  # interpolation='nearest',
                  interpolation='none',
                  cmap='binary',
                  vmin=0, vmax=16)
    # plt.text(-8, 3, "y = %.2f" % y[i])

    imagem_digito.set_xticks(())
    imagem_digito.set_yticks(())

plt.show()

# ------------------------------------------------------------------------------
#  Histograma dos dados
#  Eixo vertical: Número de instâncias
#  Eixo horizontal: Determinado intervalo valores
# ------------------------------------------------------------------------------

dados_digits.hist(bins=50, figsize=(35, 35))
plt.show()

# ------------------------------------------------------------------------------
# Embaralhar o conjunto de dados para garantir que a divisão entre os dados de
# treino e os dados de teste esteja isenta de qualquer viés de seleção
# ------------------------------------------------------------------------------

dados_embaralhados = dados_digits.sample(frac=1, random_state=11012005)

# ------------------------------------------------------------------------------
# Criar os arrays de atributos e rotulos para um conjunto de treino e de teste
# iniciais (que seram utilizados para treinar o modelo) e um outro conjunto de
# treino e teste para o teste final do modelo
# ------------------------------------------------------------------------------

atributos_teste_inicial = dados_embaralhados.iloc[:1297, :-1].to_numpy()   # ou :-1].values
rotulos_teste_inicial = dados_embaralhados.iloc[:1297, -1].to_numpy()    # ou -1].values

atributos_teste_final = dados_embaralhados.iloc[1297:, :-1].to_numpy()    # ou :-1].values
rotulos_teste_final = dados_embaralhados.iloc[1297:, -1].to_numpy()       # ou -1].values

atributos_treino, atributos_teste, rotulos_treino, rotulos_teste = train_test_split(atributos_teste_inicial,
                                                                                    rotulos_teste_inicial,
                                                                                    test_size=0.35,
                                                                                    random_state=11012005)

# -------------------------------------------------------------------------------
# Treinar um classificador SGD com o conjunto de treino
# -------------------------------------------------------------------------------

classificador_sgd = SGDClassifier(random_state=11012005)
classificador_sgd.fit(atributos_treino, rotulos_treino)

# -------------------------------------------------------------------------------
# Obter as respostas do classificador no mesmo conjunto onde foi treinado
# -------------------------------------------------------------------------------

rotulos_resposta_treino = classificador_sgd.predict(atributos_treino)

# -------------------------------------------------------------------------------
# Obter as respostas do classificador no conjunto de teste
# -------------------------------------------------------------------------------

rotulos_resposta_teste = classificador_sgd.predict(atributos_teste)

# -------------------------------------------------------------------------------
# Verificar a acurácia do classificador
# -------------------------------------------------------------------------------

print("\n\t-----Classificador SGD (Dentro da amostra)-----\n")

total = len(rotulos_treino)
acertos = sum(rotulos_resposta_treino == rotulos_treino)
erros = sum(rotulos_resposta_treino != rotulos_treino)

print("Total de amostras: ", total)
print("Respostas corretas:", acertos)
print("Respostas erradas: ", erros)

acuracia = acertos / total

print("Acurácia = %.1f %%" % (100*acuracia))
print("Taxa Erro = %4.1f %%" % (100*(1-acuracia)))

print("\n\t-----Classificador SGD (Fora da amostra)-----\n")

total = len(rotulos_teste)
acertos = sum(rotulos_resposta_teste == rotulos_teste)
erros = sum(rotulos_resposta_teste != rotulos_teste)

print("Total de amostras: ", total)
print("Respostas corretas:", acertos)
print("Respostas erradas: ", erros)

acuracia = acertos / total

print("Acurácia  = %4.1f %%" % (100*acuracia))
print("Taxa Erro = %4.1f %%" % (100*(1-acuracia)))

# -------------------------------------------------------------------------------
# Matriz de confusão
# -------------------------------------------------------------------------------

matriz_de_confusao = confusion_matrix(rotulos_resposta_teste, rotulos_teste)
print('\n\n\t-----Matriz de Confusao-----\n\n', matriz_de_confusao)

# -------------------------------------------------------------------------------
# Verificar os erros cometidos pelo classificador
# -------------------------------------------------------------------------------

# Armazena a lista com as amostras dos erros
indice_erro = np.where(rotulos_resposta_teste != rotulos_teste)[0]
quantidade_erros = len(indice_erro)

# Exibe uma figura com os digitos dos erros.
plt.figure(figsize=(30, 30))
for i in range(quantidade_erros):

    if quantidade_erros % 2 == 0:
        quantidade_linhas_figura = 2
        quantidade_colunas_figura = int(quantidade_erros / 2)

    elif quantidade_erros % 3 == 0:
        quantidade_linhas_figura = 3
        quantidade_colunas_figura = int(quantidade_erros / 3)

    elif quantidade_erros % 5 == 0:
        quantidade_linhas_figura = 5
        quantidade_colunas_figura = int(quantidade_erros / 5)

    else:
        quantidade_linhas_figura = 3
        quantidade_colunas_figura = 10

    plot_erro = plt.subplot(quantidade_linhas_figura, quantidade_colunas_figura, i + 1)
    plot_erro.set_title("gabarito = %d ; resposta = %d" %
                        (rotulos_teste[indice_erro[i]], rotulos_resposta_teste[indice_erro[i]]))

    plot_erro.imshow(atributos_teste[indice_erro[i], :].reshape(8, 8),
                     # interpolation='spline16',
                     interpolation='nearest',
                     cmap='binary',
                     vmin=0, vmax=16)

    plot_erro.set_xticks(())
    plot_erro.set_yticks(())

plt.show()
