import random
import numpy as np
from numpy.core.numeric import Inf
import pandas as pd
import math

random.seed(2)
np.random.seed(2)

def ativacao(x):
    if x > 10:
        x = (1.0-math.exp(-10))/(1.0+math.exp(-10))
    elif x < -5:
        x = (1.0-math.exp(5))/(1.0+math.exp(5))
    else :
        x = (1.0-math.exp(-x))/(1.0+math.exp(-x))
    return x

def der_ativacao(x):
    if x > 10:
        x = (2.0*math.exp(-10))/((1.0+math.exp(-10))**2)
    elif x < -5:
        x = (2.0*math.exp(5))/((1.0+math.exp(5))**2)
    else:
        x = (2.0*math.exp(-x))/((1.0+math.exp(-x))**2)
    return x

def calc_erro(targets, outputs):
    return np.sum((outputs - targets)**2)

def atualiza_pesos(pesos, erro, in_out, der_ativacao_v, alpha):
    peso_atualizado = pesos - (alpha * erro * der_ativacao_v * in_out)
    return peso_atualizado

def inicializa_pesos(lin, col):
    return np.random.rand(lin, col)[0]

def faz_a_conta(numerador_treino, denominador_treino, entrada_treino, saida_treino, erro, qtd_coef_num, qtd_coef_den, ErroMinimo):
    if(np.count_nonzero(entrada_treino == 0) < np.count_nonzero(saida_treino == 0)+qtd_coef_num-1):
        add_zeros = np.count_nonzero(saida_treino == 0)+qtd_coef_num-1-np.count_nonzero(entrada_treino == 0)
        entrada_treino = np.append(np.zeros((1, add_zeros)), entrada_treino)
    numerador_treino   = numerador_treino[qtd_coef_num, qtd_coef_den, 0:qtd_coef_num]
    denominador_treino = denominador_treino[qtd_coef_num, qtd_coef_den, 0:qtd_coef_den]
    net  = np.zeros((1, qtd_coef_den))
    erro = erro[qtd_coef_num, qtd_coef_den]
    for l in range(saida_treino.size):
        net = np.append(net, np.sum(numerador_treino * entrada_treino[l:qtd_coef_num+l]) + np.sum(denominador_treino * net[l:qtd_coef_den+l]))
        if abs(net[-1] - saida_treino[l]) > 1000:
            break
        y_net   = ativacao(net[-1])
        saida_a = ativacao(saida_treino[l])
        erro_atual = y_net - saida_a
        erro       = np.append(erro, erro_atual**2)
        der_ativacao_v = der_ativacao(net[-1])
        
        numerador_treino   = atualiza_pesos(numerador_treino, erro_atual, der_ativacao_v, entrada_treino[l:qtd_coef_num+l], alpha)
        denominador_treino = atualiza_pesos(denominador_treino, erro_atual, der_ativacao_v, net[l:qtd_coef_den+l], alpha)
    erro = 0.5*np.sum(erro)
    return np.append(numerador_treino, np.zeros((1, 4-qtd_coef_num))), np.append(denominador_treino, np.zeros((1, 3-qtd_coef_den))), erro

def testa_rede(numerador_treino, denominador_treino, entrada_teste, saida_teste, qtd_coef_num, qtd_coef_den):
    if(np.count_nonzero(entrada_teste[0:np.where(entrada_teste != 0)[0][0]] == 0) < np.count_nonzero(saida_teste == 0)+qtd_coef_num-1):
        add_zeros = np.count_nonzero(saida_teste == 0)+qtd_coef_num-1-np.count_nonzero(entrada_teste[0:np.where(entrada_teste != 0)[0][0]] == 0)
        entrada_teste = np.append(np.zeros((1, add_zeros)), entrada_teste)
    net  = np.zeros((1, qtd_coef_den))
    erro = np.zeros(1)
    numerador_treino   = numerador_treino[qtd_coef_num, qtd_coef_den, 0:qtd_coef_num]
    denominador_treino = denominador_treino[qtd_coef_num, qtd_coef_den, 0:qtd_coef_den]
    for l in range(saida_teste.size):
        net        = np.append(net, np.sum(numerador_treino * entrada_teste[l:qtd_coef_num+l]) + np.sum(denominador_treino * net[l:qtd_coef_den+l]))
        erro_atual = net[-1] - saida_teste[l]
        if abs(net[-1] - saida_teste[l]) > 1000:
            erro = np.append(erro, 1000**2)
        else:
            erro = np.append(erro, erro_atual**2)
    erro = 0.5*np.sum(erro)
    return erro


if __name__ == "__main__":
    dados = pd.DataFrame()
    for qtd_coef_num in range(1, 5):
        for qtd_coef_den in range(1, 4):
            if((qtd_coef_num == 3 and qtd_coef_den == 1) or (qtd_coef_num == 4 and qtd_coef_den != 3)):
                continue

            # Carrega os dados
            dados = dados.append(pd.read_csv('./Data_sets/Data_set_10_{}_{}_tst.csv'.format(qtd_coef_num, qtd_coef_den), sep=';', header=None, na_filter=True, index_col=False))
    
    n_amostras_teste = 100
    idx_teste = np.random.randint(0, 900, n_amostras_teste)

    #Inicializa as variáveis globais
    alpha      = 0.1
    iterations = 14000
    erro_global, iterations_global, erro_teste = np.zeros((5, 4, 1)), np.zeros((5, 4, 1)), np.zeros((5, 4, 3))
    numerador_treino, denominador_treino = np.zeros((5, 4, 4)), np.zeros((5, 4, 3))
    rede_ideal, rede_obtida, rede_obtida_iter, posicao_iter, erro_teste_global = np.empty(n_amostras_teste, dtype=object), np.empty(n_amostras_teste, dtype=object), np.empty(n_amostras_teste, dtype=object), np.empty(n_amostras_teste, dtype=object), np.empty(n_amostras_teste, dtype=object)
    erro_rede_ideal, erro_rede_obtida, iter_rede_obtida, iter_rede_ideal, posicao_erro = np.zeros(n_amostras_teste), np.zeros(n_amostras_teste), np.zeros(n_amostras_teste), np.zeros(n_amostras_teste), np.zeros(n_amostras_teste)

    numeradores, denominadores = np.zeros((n_amostras_teste, 4)), np.zeros((n_amostras_teste, 4))
    j = 0
    ErroMinimo = 0.000001
    VariacaoEM = 1000
    offset = 25
    dict_ft = {
        "5" : "1x1",
        "6" : "1x2",
        "7" : "1x3",
        "9" : "2x1",
        "10" : "2x2",
        "11" : "2x3",
        "14" : "3x2",
        "15" : "3x3",
        "19" : "4x3"
    }
    for i in idx_teste:
        if i != 0:
            i = i*10
        if j < offset-25:
            j += 1
            continue
        # Seleciona um data_set para alimentar a rede
        numerador      = dados[i:i+1].dropna(axis='columns', how='all').to_numpy()[0:1][0]
        denominador    = dados[i+1:i+2].dropna(axis='columns', how='all').to_numpy()[0:1][0]
        entrada_treino = dados[i+2:i+3].dropna(axis='columns', how='all').to_numpy()[0:1][0]
        saida_treino   = dados[i+3:i+4].dropna(axis='columns', how='all').to_numpy()[0:1][0]
        entrada_treino = np.append(np.zeros(np.count_nonzero(saida_treino == 0)-1), entrada_treino)
        entrada_teste1 = dados[i+4:i+5].dropna(axis='columns', how='all').to_numpy()[0:1][0]
        saida_teste1   = dados[i+5:i+6].dropna(axis='columns', how='all').to_numpy()[0:1][0]
        entrada_teste1 = np.append(np.zeros(np.count_nonzero(saida_teste1 == 0)-1), entrada_teste1)
        entrada_teste2 = dados[i+6:i+7].dropna(axis='columns', how='all').to_numpy()[0:1][0]
        saida_teste2   = dados[i+7:i+8].dropna(axis='columns', how='all').to_numpy()[0:1][0]
        entrada_teste2 = np.append(np.zeros(np.count_nonzero(saida_teste2 == 0)-1), entrada_teste2)
        entrada_teste3 = dados[i+8:i+9].dropna(axis='columns', how='all').to_numpy()[0:1][0]
        saida_teste3   = dados[i+9:i+10].dropna(axis='columns', how='all').to_numpy()[0:1][0]
        entrada_teste3 = np.append(np.zeros(np.count_nonzero(saida_teste3 == 0)-1), entrada_teste3)
        
        erro, iterations_when_break = np.zeros((5, 4, 1))+1000000, np.zeros((5, 4, 1))+1000000
        print(j)
        print("numerador: ", numerador,  "   denominador: ", denominador)
        for qtd_coef_num in range(1, 5):
            for qtd_coef_den in range(1, 4):
                if((qtd_coef_num == 3 and qtd_coef_den == 1) or (qtd_coef_num == 4 and qtd_coef_den != 3)):
                    continue
                erro[qtd_coef_num, qtd_coef_den] = 0
                ultimos_erros = np.zeros(10)

                #Inicializa os pesos
                numerador_treino[qtd_coef_num, qtd_coef_den]   = np.append(inicializa_pesos(1,qtd_coef_num), np.zeros((1,4-qtd_coef_num)))
                denominador_treino[qtd_coef_num, qtd_coef_den] = np.append(inicializa_pesos(1,qtd_coef_den), np.zeros((1,3-qtd_coef_den)))
                if j < 33:
                    continue 
                for k in range(iterations):
                    numerador_treino[qtd_coef_num, qtd_coef_den], denominador_treino[qtd_coef_num, qtd_coef_den], erro[qtd_coef_num, qtd_coef_den] = faz_a_conta(numerador_treino, denominador_treino, entrada_treino, saida_treino, erro, qtd_coef_num, qtd_coef_den, ErroMinimo)
                    ultimos_erros[k%10] = erro[qtd_coef_num, qtd_coef_den]
                    if erro[qtd_coef_num, qtd_coef_den] <= ErroMinimo or \
                    (erro[qtd_coef_num, qtd_coef_den] > 0.5 and k >= iterations/3) or \
                    (erro[qtd_coef_num, qtd_coef_den] > 0.1 and k >= 2*iterations/3 ) or \
                    (k%10 == 0 and abs(ultimos_erros[0]-ultimos_erros[9]) < ErroMinimo/VariacaoEM):
                        break
                iterations_when_break[qtd_coef_num, qtd_coef_den] = k+1
                
                erro_teste[qtd_coef_num, qtd_coef_den][0] = testa_rede(numerador_treino, denominador_treino, entrada_teste1, saida_teste1, qtd_coef_num, qtd_coef_den)
                erro_teste[qtd_coef_num, qtd_coef_den][1] = testa_rede(numerador_treino, denominador_treino, entrada_teste2, saida_teste2, qtd_coef_num, qtd_coef_den)
                erro_teste[qtd_coef_num, qtd_coef_den][2] = testa_rede(numerador_treino, denominador_treino, entrada_teste3, saida_teste3, qtd_coef_num, qtd_coef_den)
                # print("Erro: ",  erro_teste[qtd_coef_num, qtd_coef_den])
        if j < 33:
            j += 1
            continue
        # Guarda as metrícas para analisar a eficácia da rede posteriormente
        erro_global       = np.c_[erro_global, erro]
        iterations_global = np.c_[iterations_global, iterations_when_break]
        count_num = np.count_nonzero(numerador)
        count_den = np.count_nonzero(denominador)-1
        ordena_erros        = [erro[1,1], erro[1,2], erro[1,3], erro[2,1], erro[2,2], erro[2,3],
                               erro[3,2], erro[3,3], erro[4,3]]
        ordena_iter         = [iterations_when_break[1,1], iterations_when_break[1,2], iterations_when_break[1,3], 
                               iterations_when_break[2,1], iterations_when_break[2,2], iterations_when_break[2,3],
                               iterations_when_break[3,2], iterations_when_break[3,3], iterations_when_break[4,3]]
        rede_ideal[j]       = ("{}x{}").format(count_num,count_den)
        erro_rede_ideal[j]  = erro[count_num, count_den]
        posicao_erro[j]     = np.where(sorted(ordena_erros) == erro_rede_ideal[j])[0]+1
        rede_obtida[j]      = dict_ft["{}".format(np.nanargmin(erro))]
        rede_obtida_iter[j] = dict_ft["{}".format(np.nanargmin(iterations_when_break))]
        iter_rede_ideal[j]  = iterations_when_break[count_num, count_den]
        posicao_iter_un     = np.where(ordena_iter == iter_rede_ideal[j])[0]
        if len(posicao_iter_un) > 1:
            erros = np.array([10000])
            for i in posicao_iter_un:
                erros = np.append(erros, ordena_erros[i])
            posicao_erro_iter = np.where(sorted(erros) == erro_rede_ideal[j])[0]
            posicao_iter[j] = np.where(sorted(ordena_iter) == iter_rede_ideal[j])[0][posicao_erro_iter]+1
        else :
            posicao_iter[j] = posicao_iter_un[0]+1
        iter_rede_obtida[j] = iterations_when_break.min()
        num_den = dict_ft["{}".format(np.nanargmin(erro))].split('x')
        erro_teste_global[j] = [erro_teste[count_num, count_den], erro_teste[int(num_den[0]), int(num_den[1])]]
        erro_rede_obtida[j]  = erro[int(num_den[0]), int(num_den[1])]
        numeradores[j]   = np.append(np.zeros((1, 4-numerador.size)), numerador)
        denominadores[j] = np.append(np.zeros((1, 4-denominador.size)), denominador)

        j += 1
        # metricas = "Erro medio: {}    Erro des_pad: {}    Iteracoes media: {}    Iteracoes des_pad: {}"
        # metricas = metricas.format(np.mean(erro_global), np.std(erro_global), np.mean(iterations_global), np.std(iterations_global))
        # f = open("./Metricas/metricas_all_tst.txt", "w")
        # f.write(metricas)
        # f.close()
        metricas_to_save        = np.asarray([rede_ideal, erro_rede_ideal, iter_rede_ideal, posicao_erro, rede_obtida, erro_rede_obtida, rede_obtida_iter, iter_rede_obtida, posicao_iter, erro_teste_global], dtype=object)
        metricas_to_save_global = np.asarray([erro_global, iterations_global, numeradores, denominadores], dtype=object)
        offset = 33
        pd.DataFrame(metricas_to_save).to_csv("./Metricas/metricas_all_{}_10_tst_EM{}_V{}.csv".format(offset, ErroMinimo, VariacaoEM), sep=";")
        pd.DataFrame(metricas_to_save_global).to_csv("./Metricas/metricas_all_g_{}_10_tst_EM{}_V{}.csv".format(offset, ErroMinimo, VariacaoEM), sep=";")
        offset = 25
        if j == offset+75:
            break
    # print(metricas)