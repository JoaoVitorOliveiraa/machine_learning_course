#==============================================================================
# EXPERIMENTO 01 - Explorar e visualizar o conjunto de dados IRIS
#==============================================================================

#------------------------------------------------------------------------------
# Importar a biblioteca Pandas, PyPlot e Path
#------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#------------------------------------------------------------------------------
# Carregar o conjunto de dados IRIS do arquivo CSV
#------------------------------------------------------------------------------

caminho_dados_iris = Path('../data') / 'Iris_Data.csv'
dados_iris = pd.read_csv(caminho_dados_iris, delimiter=',', decimal='.')

#dados = pd.read_csv('C:/Users/Pichau/Documents/João/UFRJ/Matérias/Quinto Período/Introdução ao Aprendizado de Máquina/scripts/Iris_Data.csv', delimiter=',', decimal='.')
# dados = pd.read_csv('Iris_Data.csv', delimiter=',', decimal='.')

#------------------------------------------------------------------------------
# Exibir informações sobre o conjunto de dados
#------------------------------------------------------------------------------

print("\n\n Exibir as 4 primeiras amostras: \n")
print(dados_iris.head(n=4))


print("\n\n Exibir as 3 últimas amostras: \n")
print(dados_iris.tail(n=3))

print("\n\n Exibir as dimensões do conjunto de dados: \n")
print(dados_iris.shape)
print("O conjunto tem",dados_iris.shape[0],"amostras com",dados_iris.shape[1],"variáveis")

print("\n\n Exibir os tipos das variáveis do conjunto de dados: \n")
print(dados_iris.dtypes)

print("\n\n Exibir as 5 primeiras amostras de um dataframe somente com dados das pétalas: \n")
dados_das_petalas = dados_iris[ ['petal_length','petal_width'] ]
print(dados_das_petalas.head())

print("\n\n Retirar o prefixo 'Iris-' do nome da espécie: \n")
dados_iris['species'] = dados_iris['species'].str.replace('Iris-','')
print(dados_iris.head())

print("\n\n Outra forma (mais flexível) de retirar o prefixo 'Iris-': \n")
dados_iris['species'] = dados_iris['species'].apply(lambda r: r.replace('Iris-',''))
print(dados_iris.head())

print("\n\n Contabilizar a quantidade de amostras de cada espécie: \n")
print(dados_iris['species'].value_counts())

print("\n\n Exibir informações estatísticas sobre os dados: \n")
print(dados_iris.describe())

print("\n\n Exibir a média de cada coluna: \n")
print(dados_iris.iloc[:,:-1].mean())

print("\n\n Exibir a mediana de cada coluna: \n")
print(dados_iris.iloc[:,:-1].median())

print("\n\n Exibir desvio-padrão de cada coluna: \n")
print(dados_iris.iloc[:,:-1].std())

print("\n\n Exibir a média de cada coluna por espécie: \n")
print(dados_iris.groupby('species').mean())

print("\n\n Exibir o desvio-padrão de cada coluna por espécie: \n")
print(dados_iris.groupby('species').std())

print("\n\n Montar tabela com informações estatísticas personalizadas: \n")
resultado = dados_iris.groupby('species').agg(
    {
     'petal_length': ['median','mean','std'],
     'petal_width' : ['median','mean','std']
     }
    )
print(resultado)

print("\n\n Montar tabela com informações estatísticas de todas os atributos: \n")
resultado = dados_iris.groupby('species').agg(
    {
     x: ['mean','std'] for x in dados_iris.columns if x != 'species'
     }
    )
print(resultado.to_string())

#------------------------------------------------------------------------------
# Exibir gráficos
#------------------------------------------------------------------------------

print("\n\n Visualizar o histograma de uma variável: \n")

grafico = dados_iris['petal_length'].plot.hist(bins=30)

grafico.set(
    title  = 'DISTRIBUIÇÃO DO COMPRIMENTO DA PÉTALA',
    xlabel = 'Comprimento da Pétala (cm)',
    ylabel = 'Número de amostras'
    )

plt.show()


print("\n\n Visualizar o diagrama de dispersão entre duas variável: \n")

grafico = dados_iris.plot.scatter('petal_width','petal_length')

grafico.set(
    title  = 'DISPERSÃO LARGURA vs COMPRIMENTO DA PÉTALA',
    xlabel = 'Largura da Pétala (cm)',
    ylabel = 'Comprimento da Pétala (cm)'
    )

plt.show()

#------------------------------------------------------------------------------
# Separar os atributos e o alvo em dataframes distintos
#------------------------------------------------------------------------------

atributos = dados_iris.iloc[:,:4]
rotulos = dados_iris.iloc[:,4]

#------------------------------------------------------------------------------
# Fazer a mesma coisa com uso de índice negativo
#   - os atributos são todas as colunas exceto a última
#   - os rótulos estão na última coluna
#------------------------------------------------------------------------------

atributos = dados_iris.iloc[:,:-1]
rotulos = dados_iris.iloc[:,-1]

#------------------------------------------------------------------------------
# Montar lista com os valores distintos dos rótulos (classes)
#------------------------------------------------------------------------------

# unique(): Retorna um array com os valores únicos presentes na coluna "species".
# tolist(): Converte o array retornado por ".unique()" em uma lista Python.
classes = dados_iris['species'].unique().tolist()
print("Classes: ", classes)

#------------------------------------------------------------------------------
# Montar mapa de cores associando cada classe a uma cor
#------------------------------------------------------------------------------

mapa_de_cores = ['red','green','blue']
cores_das_amostras = [mapa_de_cores[classes.index(r)] for r in rotulos]

#------------------------------------------------------------------------------
# Visualizar a matriz de dispersão dos atributos
#------------------------------------------------------------------------------

pd.plotting.scatter_matrix(
    atributos,
    c=cores_das_amostras,
    figsize=(12,12),
    marker='o',
    s=50,
    alpha=0.50,
    diagonal='hist',         # 'hist' ou 'kde'
    hist_kwds={'bins':20}
    )

plt.suptitle(
    'MATRIZ DE DISPERSÃO DOS ATRIBUTOS',
    y=0.9,
    fontsize='xx-large'
    )

plt.show()  # Exibe o gráfico

#------------------------------------------------------------------------------
# Visualizar um gráfico de dispersão 3D entre 3 atributos
#------------------------------------------------------------------------------

# Escolher as variáveis de cada eixo
eixo_x = 'sepal_length'
eixo_y = 'petal_length'
eixo_z = 'petal_width'

# Criar uma figura
figura = plt.figure(figsize=(18,15))

# Criar um grafico 3D dentro da figura
grafico = figura.add_subplot(111, projection='3d')

# Plotar o diagrama de dispersão 3D
grafico.scatter(
    dados_iris[eixo_x],
    dados_iris[eixo_y],
    dados_iris[eixo_z],
    c=cores_das_amostras,
    marker='o',
    s=50,
    alpha=0.5
    )

plt.suptitle(
    'GRÁFICO DE DISPERSÃO 3D ENTRE 3 VARIÁVEIS',
    y=0.85,
    fontsize='xx-large'
    )

grafico.set_xlabel(eixo_x, fontsize='xx-large')
grafico.set_ylabel(eixo_y, fontsize='xx-large')
grafico.set_zlabel(eixo_z, fontsize='xx-large')

plt.show()  # Exibe o gráfico