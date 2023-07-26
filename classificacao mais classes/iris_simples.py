import pandas as pd

# modelo sequential
from keras.models import Sequential

# modelo fully connected (todos os neurônios fazem ligação com o próximo)
from keras.layers import Dense
from keras.utils import np_utils

base = pd.read_csv('iris.csv')

# iloc é uma funcao que irá fazer a divisao dos valores, ou seja, ele irá receber apenas os previsores
# : = irá pegar todos os valores (todas as linhas)
# 0:4 = do atributo 0 até o 4
# values = converter no formato do numpy
previsores = base.iloc[:, 0:4].values

classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder

# funcao que faz conversao de formato categorico para numerico
from sklearn.model_selection import train_test_split

# jogar o metodo para uma variavel
Labelencoder = LabelEncoder()

# transformador em si
classe = Labelencoder.fit_transform(classe)

# vai receber a classe (saida) separada devidamente para a execucao sem erro
# onde estamos trabalhando com 3 dimensões
classe_dummy = np_utils.to_categorical(classe)
# iris setosa     1 0 0
# iris virginica  0 1 0
# iris versicolor 0 0 1

# faz a divisao da base de teste e treinamento e o teste_size vai definir a porcentagem de separação
# no caso 75%/25%
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

# iniciar a rede
classificador = Sequential()

# construção da estrutara da rede neural
# adicionar as camadas

# units = quantidade de neurônios
# activation = funcao de ativacao
# input_dim = quantos atributos possuem na camada de entrada (só precisa informar na primeira
# camada escondida)
# camada escondida
classificador.add(Dense(units= 4, activation= 'relu', input_dim = 4))

# segunda camada oculta
classificador.add(Dense(units= 4, activation= 'relu'))

# camada de saída
# softmax = usado para problemas com classificação com mais de 2 classes (2 saídas)
classificador.add(Dense(units= 3, activation= 'softmax'))

# compilacao da rede neural
# loss function vai ser categorial cross por serem mais de 2 saídas
# o mesmo vale para o metrics
classificador.compile(optimizer= 'adam', loss= 'categorical_crossentropy',
                      metrics= ['categorical_accuracy'])

# treinamento
classificador.fit(previsores_treinamento, classe_treinamento, batch_size= 10,
                  epochs = 1000)

# avaliacao do resultado
# linhas:
# loss function
# porcentagem de acerto
resultado = classificador.evaluate(previsores_teste, classe_teste)

# resultados da tabela teste
previsoes = classificador.predict(previsores_teste)

# retornar full or false
previsoes = (previsoes > 0.5)

import numpy as np

# transforma a classe_teste que tem 3 indices em apenas 1 entrada que retorna
# o maior indice
classe_teste2 = [np.argmax(t) for t in classe_teste]

previsoes2 = [np.argmax(t) for t in previsoes]

# importacao da matriz de confusao
# visualizacao de qual classe está acertando mais e qual está errando mais
from sklearn.metrics import confusion_matrix

matriz = confusion_matrix(previsoes2, classe_teste2)