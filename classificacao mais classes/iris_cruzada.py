import pandas as pd

# modelo sequential
from keras.models import Sequential

# modelo fully connected (todos os neurônios fazem ligação com o próximo)
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('iris.csv')

# iloc é uma funcao que irá fazer a divisao dos valores, ou seja, ele irá receber apenas os previsores
# : = irá pegar todos os valores (todas as linhas)
# 0:4 = do atributo 0 até o 4
# values = converter no formato do numpy
previsores = base.iloc[:, 0:4].values

classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder

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

def criar_rede():
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
    
    return classificador

# inicializa a rede com as epocas e o batch_size
classificador = KerasClassifier(build_fn= criar_rede,
                                epochs= 1000,
                                batch_size= 10)

# cross_validation
# cv = divisoes que o cross validation irá fazer
resultados = cross_val_score(estimator= classificador,
                             X= previsores,
                             y= classe,
                             cv= 10,
                             scoring= 'accuracy')

# media dos 10 resultados
media = resultados.mean()

# verificar se o valor está variando muito para evitar overfitting
desvio = resultados.std()

# fazer mais testes com os parâmetros da rede neural para melhorar o resultado