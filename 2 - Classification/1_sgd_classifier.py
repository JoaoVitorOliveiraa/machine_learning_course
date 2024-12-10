#==============================================================================
# Tópico 01 - Treinando um Classificador Binário com a função SGDClassifier
#==============================================================================

#------------------------------------------------------------------------------
# Importar a biblioteca Pandas, Pyplot e Sklearn
#------------------------------------------------------------------------------

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

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
#  Visualizar o primeiro digito
# ------------------------------------------------------------------------------

# Cada imagem tem 8 x 8 pixels e cada característica representa a intensidade do
# pixel, do 0 (branco) a 16 (preto).

primeiro_digito = atributos[0]
plt.figure(figsize=(8, 8))
imagem_primeiro_digito = plt.subplot(111)
imagem_primeiro_digito.set_title("Rótulo = %.0f" % rotulos[0])
imagem_primeiro_digito.imshow(primeiro_digito.reshape(8, 8), cmap="binary")
imagem_primeiro_digito.set_xticks(())
imagem_primeiro_digito.set_yticks(())
plt.show()

# ------------------------------------------------------------------------------
# Embaralhar o conjunto de dados para garantir que a divisão entre os dados de
# treino e os dados de teste esteja isenta de qualquer viés de seleção
# ------------------------------------------------------------------------------

dados_embaralhados = dados_digits.sample(frac=1, random_state=11012005)

# ------------------------------------------------------------------------------
# Criar os arrays de atributos e rotulos para um conjunto de treino e de teste
# ------------------------------------------------------------------------------


atributos_treino, atributos_teste, rotulos_treino, rotulos_teste = train_test_split(atributos,
                                                                                    rotulos,
                                                                                    test_size=0.25,
                                                                                    random_state=11012005)

# ------------------------------------------------------------------------------
#  Transformando os rótulos de treino e teste para o classificador binário
#  Somente será capaz de fazer distinções entre apenas duas classes, 0 e não 0
# ------------------------------------------------------------------------------

rotulos_treino_0 = (rotulos_treino==0)  # True para todos os 0s, False para todos os outros
rotulos_teste_0 = (rotulos_teste==0)

# -------------------------------------------------------------------------------
# Treinar um classificador SGD com o conjunto de treino
# -------------------------------------------------------------------------------

classificador_sgd = SGDClassifier(random_state=11012005)
#classificador_sgd.fit(atributos_treino, rotulos_treino_0); Treinamos abaixo com a validação cruzada

# -------------------------------------------------------------------------------
# Obtendo a acurácia dos 3 folds da validação cruzada
# -------------------------------------------------------------------------------

acuracias_cross = cross_val_score(classificador_sgd, atributos_treino, rotulos_treino_0, cv=3, scoring='accuracy')

print("\n\n\t-----Acurácias da Validação Cruzada-----\n")
for index in range(len(acuracias_cross)):
    print(f"Acurácia {index+1}: {(100 * acuracias_cross[index]):.3f}%")

print(f"Médias das Acurácias: {(100 * acuracias_cross.mean()):.3f}%")

# -------------------------------------------------------------------------------
# Predição em cada um dos folds
# -------------------------------------------------------------------------------

rotulos_resposta_treino = cross_val_predict(classificador_sgd, atributos_treino, rotulos_treino_0, cv=3)

# -------------------------------------------------------------------------------
# Matriz de confusão
# -------------------------------------------------------------------------------

matriz_de_confusao = confusion_matrix(rotulos_treino_0, rotulos_resposta_treino)
print('\n\n\t-----Matriz de Confusao-----\n\n', matriz_de_confusao)

# -------------------------------------------------------------------------------
# Obtendo a precisão através da função precision_score()
# -------------------------------------------------------------------------------

precisao = precision_score(rotulos_treino_0, rotulos_resposta_treino)
print(f'\n\n\t-----Precisão da Predição-----\n\nPrecisão: {(100*precisao):.3f}')

# -------------------------------------------------------------------------------
# Obtendo a revocação através da função recall_score()
# -------------------------------------------------------------------------------

revocacao = recall_score(rotulos_treino_0, rotulos_resposta_treino)
print(f'\n\n\t-----Revocação da Predição-----\n\nRevocação: {(100*revocacao):.3f}')

# -------------------------------------------------------------------------------
# Calculo da métrica F1 através da função recall_score()
# "average" default, pois o classificador é binário
# -------------------------------------------------------------------------------

metrica_f1 = f1_score(rotulos_treino_0, rotulos_resposta_treino)
print(f"\n\n\t-----F1 da Predição-----\n\nF1: {(100*metrica_f1):.3f}%")

# -------------------------------------------------------------------------------
# Obtendo os scores de decisão de todas as instâncias no conjunto de treinamento
# "decision_function()" retorna um score para cada instância e, em seguida, faz as
# predições com base nesses scores usando qualquer limiar desejado
# -------------------------------------------------------------------------------

scores_decisao = cross_val_predict(classificador_sgd, atributos_treino, rotulos_treino_0, cv=3, method='decision_function')
print(f"\n\n\t-----Scores de decisão de todas as instâncias no conjunto de treinamento-----\n\nScores: {scores_decisao}")

# -------------------------------------------------------------------------------
# Calculando a precisão e a revocação em todos os limiares possíveis através da
# função precision_recall_curve
# -------------------------------------------------------------------------------

precisoes, revocacoes, limiares = precision_recall_curve(rotulos_treino_0, scores_decisao)

# -------------------------------------------------------------------------------
# Plotando a precisão e a revocação como funções do valor limiar
# -------------------------------------------------------------------------------

plt.plot(limiares, precisoes[:-1], "b--", label="Precisão")
plt.plot(limiares, revocacoes[:-1], "g-", label="Revocações")
plt.xlabel("Limiares")
plt.legend()
plt.show()

# -------------------------------------------------------------------------------
# Calculando o TPR (True Positives) e o FPR (False Positives) para diversos
# valores de limiares, a fim de plotar a curva ROC
# -------------------------------------------------------------------------------

falsos_positivos, verdadeiros_positivos, limiares = roc_curve(rotulos_treino_0, scores_decisao)

# -------------------------------------------------------------------------------
# Plotando a curva ROC
# -------------------------------------------------------------------------------

plt.plot(falsos_positivos, verdadeiros_positivos, linewidth=2)
plt.plot([0,1], [0,1], 'k--')       # Diagonal tracejada
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.grid()
plt.legend()
plt.show()

# -------------------------------------------------------------------------------
# Calculando a área sob a curva (AUC)
# -------------------------------------------------------------------------------

area_sob_curva = roc_auc_score(rotulos_treino_0, scores_decisao)
print(f"\n\n\t-----Área sob a curva (AUC)-----\n\nÁrea: {area_sob_curva}")