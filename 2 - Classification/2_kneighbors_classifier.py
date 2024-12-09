#==============================================================================
# Tópico 01 - Treinando um Classificador com a função KNeighborsClassifier
#==============================================================================

#------------------------------------------------------------------------------
# Importar a biblioteca Pandas, PyPlot e Path
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

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

atributos = dados_digits.iloc[:, :-1].to_numpy()   # ou :-1].values
rotulos = dados_digits.iloc[:, -1].to_numpy()      # ou :-1].values
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


atributos_treino, atributos_teste, rotulos_treino, rotulos_teste = train_test_split(atributos,
                                                                                    rotulos,
                                                                                    test_size=0.25,
                                                                                    random_state=11012005)

# -------------------------------------------------------------------------------
# Treinar um classificador SGD com o conjunto de treino
# -------------------------------------------------------------------------------

classificador_kneighbors = KNeighborsClassifier(n_neighbors=5, weights="uniform")
classificador_kneighbors.fit(atributos_treino, rotulos_treino)

# -------------------------------------------------------------------------------
# Obter as respostas do classificador no mesmo conjunto onde foi treinado
# -------------------------------------------------------------------------------

rotulos_resposta_treino = classificador_kneighbors.predict(atributos_treino)

# -------------------------------------------------------------------------------
# Obter as respostas do classificador no conjunto de teste
# -------------------------------------------------------------------------------

rotulos_resposta_teste = classificador_kneighbors.predict(atributos_teste)

# -------------------------------------------------------------------------------
# Verificar a acurácia do classificador
# -------------------------------------------------------------------------------

print("\n\t-----Classificador KNN (Dentro da amostra)-----\n")

total = len(rotulos_treino)
acertos = sum(rotulos_resposta_treino == rotulos_treino)
erros = sum(rotulos_resposta_treino != rotulos_treino)

print("\tTotal de amostras: ", total)
print("\tRespostas corretas:", acertos)
print("\tRespostas erradas: ", erros)

acuracia = acertos / total

print("\tAcurácia = %.1f %%" % (100*acuracia))
print("\tTaxa Erro = %4.1f %%" % (100*(1-acuracia)))

print("\n\t-----Classificador KNN (Fora da amostra)-----\n")

total = len(rotulos_teste)
acertos = sum(rotulos_resposta_teste == rotulos_teste)
erros = sum(rotulos_resposta_teste != rotulos_teste)

print("\tTotal de amostras: ", total)
print("\tRespostas corretas:", acertos)
print("\tRespostas erradas: ", erros)

acuracia = acertos / total

print("\tAcurácia  = %4.1f %%" % (100*acuracia))
print("\tTaxa Erro = %4.1f %%" % (100*(1-acuracia)))

# -------------------------------------------------------------------------------
# Matriz de confusão
# -------------------------------------------------------------------------------

matriz_de_confusao = confusion_matrix(rotulos_resposta_teste, rotulos_teste)
print('\n\n\t-----Matriz de Confusao-----\n\n', matriz_de_confusao)

# -------------------------------------------------------------------------------
# Calculo da precisão através da função precision_score(), usando a medida "macro"
# -------------------------------------------------------------------------------

precisao_score = precision_score(rotulos_teste, rotulos_resposta_teste, average='macro')
print("\n\n\t-----Precisão - precision_score()-----\n\n\tPrecisão: ", precisao_score)

# -------------------------------------------------------------------------------
# Calculo da acurácia através da função accuracy_score()
# -------------------------------------------------------------------------------

acuracia_score = accuracy_score(rotulos_teste, rotulos_resposta_teste)
print("\n\n\t-----Acurácia - accuracy_score()-----\n\n\tAcurácia: ", acuracia_score)

# -------------------------------------------------------------------------------
# Calculo da revocação/sensibilidade (recall/sensitivity) através da função
# recall_score(), usando a medida "macro"
# -------------------------------------------------------------------------------

revocacao_score = recall_score(rotulos_teste, rotulos_resposta_teste, average='macro')
print("\n\n\t-----Revocação/Sensibiliade - recall_score()-----\n\n\tRevocação: ", revocacao_score)

# -------------------------------------------------------------------------------
# Calculo da métrica F1 através da função recall_score(), usando a medida "macro"
# -------------------------------------------------------------------------------

metrica_f1 = f1_score(rotulos_teste, rotulos_resposta_teste, average='macro')
print("\n\n\t-----F1 - f1_score()-----\n\n\tF1: ", metrica_f1)

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

# -------------------------------------------------------------------------------
# Testando outros números de vizinhos
# -------------------------------------------------------------------------------

print("\n\n\t-----Testando outros números de vizinhos-----\n")
for k in range(1, 31):

    classificador = KNeighborsClassifier(n_neighbors=k, weights="distance")
    classificador = classificador.fit(atributos_treino, rotulos_treino)

    y_resposta_treino = classificador.predict(atributos_treino)
    y_resposta_teste = classificador.predict(atributos_teste)

    acuracia_treino = accuracy_score(rotulos_treino, y_resposta_treino)
    precisao_treino = precision_score(rotulos_treino, y_resposta_treino, average='macro')
    revocacao_treino = recall_score(rotulos_treino, y_resposta_treino, average='macro')
    f1_treino = f1_score(rotulos_treino, y_resposta_treino, average='macro')

    acuracia_teste  = accuracy_score(rotulos_teste, y_resposta_teste)
    precisao_teste = precision_score(rotulos_teste, y_resposta_teste, average='macro')
    revocacao_teste = recall_score(rotulos_teste, y_resposta_teste, average='macro')
    f1_teste = f1_score(rotulos_teste, y_resposta_teste, average='macro')

    print(
        f"\nNúmero de Vizinhos: {k}",
        f"\nAcurácia no Treino: {(100*acuracia_treino):.3f}%",
        f"\nErro no Treino: {(100*(1-acuracia_treino)):.3f}%",
        f"\nPrecisão no Treino: {(100*precisao_treino):.3f}%",
        f"\nRevocação no Treino: {(100*revocacao_treino):.3f}%",
        f"\nF1 no Treino: {(100*f1_treino):.3f}%",
        f"\nAcurácia no Teste: {(100*acuracia_teste):.3f}%",
        f"\nErro no Teste: {(100*(1-acuracia_teste)):.3f}%"
        f"\nPrecisão no Teste: {(100 * precisao_teste):.3f}%",
        f"\nRevocação no Teste: {(100 * revocacao_teste):.3f}%",
        f"\nF1 no Teste: {(100 * f1_teste):.3f}%\n"
    )