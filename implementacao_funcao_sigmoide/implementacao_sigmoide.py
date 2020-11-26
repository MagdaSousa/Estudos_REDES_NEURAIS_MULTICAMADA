# Rede Neural Multicamada
# Aprendizagem supervisionada
# -Implementação da função de ativação SIGMOID para a função XOR(característica: não linear)



import numpy as np

# Prévia para conhecimento da função que será utilizada como função de ativação de uma rede neural multicamada

def sigmoid(soma):
  "Nesta função sigmoid o valor mínimo será 0 e o máximo será 1,ela também não retorna valores negativos"
  return 1/(1 + np.exp(-soma))

#Se o X(soma) for alto o valor será aproximadamente 1
b=sigmoid(50)
print("função sigmoid",b)




a = np.exp(1)# colocar o valor 1, quer dizer que eu elevei o número de euler a 1
a  # número de euler



# Na montagem da rede neural multicamada, o treinamento será feito com base na função XOR, utilizando a função de ativação SIGMOID( y=1/(1+exp(-x)),segue os padrões de entrada e saída:
#
# x1 | x2  | Classe
# 0  | 0   | 0
# 0  | 1   | 1
# 1  | 0   | 1
# 1  | 1   | 0
# Está função tem uma característica NÃO LINEAR, quando colocamos em um gráfico, e por tanto a Step function, não seria adequada para a resolução deste caso, pois ela ficarem em um loop infinito, e não conseguiria encontrar os pesos adequados.Por isso para funções não lineares devemos procurar alternativas, como pro exemplo a função sigmóide



entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas =np.array([[0],[1],[1],[0]])

# Implementação dos pesos para a camada de entrada levando até a camada oculta:

pesos_camada_de_entrada_para_oculta =np.array([[-0.424,-0.740,-0.961],#pesos x1
                                              [0.358,-0.577,-0.469]])# pesos x2

pesos_camada_oculta_para_saida = np.array([[-0.017],
                                         [-0.893],
                                         [0.841]])


epocas=100 # Quantidade de vezes que irei passar e atualizar os pesos para refazer os cálculos

for j in range(epocas):
  camadaEntrada =entradas
  somaSinapses0= np.dot(camadaEntrada,pesos_camada_de_entrada_para_oculta)
  camadaOculta =sigmoid(somaSinapses0)#valores da primeira aplicação da função sigmoid na camada oculta
  somaSinapses1 = np.dot(camadaOculta,pesos_camada_oculta_para_saida)
  camadaSaida = sigmoid(somaSinapses1)# valores da primeira rodada de treino do neuronio
#calculo do erro: erro= respostaEsperada-respostaCalculada
  erroCamadaSaida = saidas -camadaSaida
  mediaAbsoluta =np.mean(np.abs(erroCamadaSaida))#abs, pegar números absolutos desconsiderando o sinal
print("sinapses",somaSinapses0)
print("camada oculta",camadaOculta)
print("camada Saida",camadaSaida)
print("erro Camada Saida",erroCamadaSaida)
print("media Absoluta",mediaAbsoluta)

#este resultado de média absoluta indica(levando me consideração de 0 a 100, que ela tem 50% de erros nesta rede neural, e 50% de acertos)
