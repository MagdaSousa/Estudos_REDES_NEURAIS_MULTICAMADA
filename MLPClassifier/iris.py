# Treinamento de uma rede neural com o dataSet Iris


from sklearn.neural_network import MLPClassifier
from sklearn import datasets # Utilizarei o dataset iris que já contém no pacote do sklearn

iris =datasets.load_iris()
entradas = iris.data # extraindo os dados com as carcterísticas da flores
saidas =iris.target
#
#
# verificando as características do Dataset Iris
# A classificação desse dataset leva em consideração 4 características:
# 'sepal length (cm)-> largura da sepal
# 'sepal width (cm)'-> tamanho da sepal
# 'petal length (cm)'-> largura da petala
# 'petal width (cm)'-> tamanho da petala
iris.feature_names

#
#
# Existem três classes de fores neste dataset:
#
# ['setosa', 'versicolor', 'virginica']
# Sendo que ao usar o dataset, esses nomes são 'covertidos' em números, então 0-setosa,1- versicolor e 2-virginia

iris.target_names
rede_neural =MLPClassifier(
    verbose =True, # mostrar as perdas e número de iterações
    activation='logistic',# tipo de algoritmo de ativação
    max_iter =1000,# valor máximo de iterações
    tol=0.0001,
    learning_rate_init=0.3)

# Treinando a rede neural com os dados característicos do dataset
# Iteration- número de iterações; loss->valor da perga ou erro do modelo a cada iteração
rede_neural.fit(entradas,saidas)
rede_neural.predict([[5.1,3.5,1.4,0.2]])