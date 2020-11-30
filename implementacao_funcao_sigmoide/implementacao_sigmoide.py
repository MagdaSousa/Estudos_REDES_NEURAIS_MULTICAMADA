# Rede Neural Multicamada
# Aprendizagem supervisionada
# -Implementação da função de ativação SIGMOID para a função XOR(característica: não linear)



import numpy as np

# Prévia para conhecimento da função que será utilizada como função de ativação de uma rede neural multicamada

def sigmoid(soma):
  "Nesta função sigmoid o valor mínimo será 0 e o máximo será 1,ela também não retorna valores negativos"
  return 1/(1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig *(1-sig)

# a= sigmoid(0.5)
# x= sigmoidDerivada(a)
# print("Sigmmoid Derivada",x)

#Se o X(soma) for alto o valor será aproximadamente 1
# b=sigmoid(50)
# print("função sigmoid",b)




# a= np.exp(1)# colocar o valor 1, quer dizer que eu elevei o número de euler a 1
# a  # número de euler


#
#
# Na montagem da rede neural multicamada, o treinamento será feito com base na função XOR,
# utilizando a função de ativação SIGMOID( y=1/(1+exp(-x)),segue os padrões de entrada e saída:
#
#    x1     |     x2      |     Classe
# -----------------------------------
#    0      |     0       |     0
# -----------------------------------
#    0      |     1       |     1
# -----------------------------------
#    1      |     0       |     1
# -----------------------------------
#    1      |     1       |     0
# -----------------------------------
#
#
# Está função tem uma característica NÃO LINEAR, quando colocamos em um gráfico, e por tanto a Step function,
# não seria adequada para a resolução deste caso, pois ela ficarem em um loop infinito, e não conseguiria encontrar os pesos adequados.
# Por isso para funções não lineares devemos procurar alternativas, como pro exemplo a função sigmóide
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([[0], [1], [1], [0]])  # sída o resultado que esu quero obter

# Implementação dos pesos para a camada de entrada levando até a camada oculta:

# ------ Neste caso os pesos foram colocados de maneira fixa, mas o ideal é colocar valores aleatórios
# pesos0 =np.array([[-0.424,-0.740,-0.961],#pesos x1
#                 [0.358,-0.577,-0.469]])# pesos x2

# pesos1 = np.array([[-0.017],[-0.893],[0.148]])     # ao final do processo essa variável vai conter o valor dos pessos necessários para se obter o valor da saída desejada

# ---- colocando valores aleatórios nos pesos:
pesos0 = 2 * np.random.random((2, 3)) - 1  # -1, para gerar valores negativos,
# e 2* foi para gerar valores, tanto negativos quanto positivos. O (2,3) indica 2 neurônios na camada de entrada e 3 na oculta
pesos1 = 2 * np.random.random((3, 1))  # (3,1)# três neurônios na camada oculta e 1 na de saída

taxaDeAprendizagem = 0.6
momento = 1
epocas = 100000  # Quantidade de vezes que irei passar e atualizar os pesos para refazer os cálculos

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapses0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapses0)  # valores da primeira aplicação da função sigmoid na camada oculta

    somaSinapses1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(
        somaSinapses1)  # valores da primeira rodada de treino do neuronio, ou seja resultado do neurônio depois de treinado
    # calculo do erro: erro= respostaEsperada-respostaCalculada
    erroCamadaSaida = saidas - camadaSaida

    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))  # abs, pegar números absolutos desconsiderando o sinal
    print("Erro---------->", str(mediaAbsoluta))  # percentual de erro, 100 - esse valor voc~e terá sua taxa de acerto

    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida  # calculo do delta para a camada de saída
    # deltaSaidaXPeso = deltaSaida.dot(pesos_camada_oculta_para_saida) # Neste caso a lista pesos_camada_oculta_para_saida, é uma matriz de 3 linhas e 1 coluna,
    # e delta saída é de 4 linhas e 1 coluna. Como elas tem quantidades diferentes não é possivel fazer o calculo desta maneira, deverá ser transformados
    # em matrizes compostas

    pesos1Transposta = pesos1.T  # Após está transformação esta matriz passará a ser 1x3, e seus pesos serão multiplicados,
    # por cada delta

    # refazendo a operações :
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)

    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaDeAprendizagem)

    camadaEntradaTransposta = camadaEntrada.T
    pesoNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesoNovo0 * taxaDeAprendizagem)

print("Resultado que eu quero obter---->", saidas)
print("resultao obtido pela rede neural---->", camadaSaida)

