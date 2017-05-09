#!/usr/bin/python
# -*- coding: utf-8 -*-

from glob import glob
import os

from sklearn.metrics.cluster import adjusted_rand_score

import numpy as np
import pandas as pd

def obter_numeros_particoes(particoes, nomes_particoes):
    numeros_particoes = []
    for i, particao_i in particoes.iterrows():
        if (encuntar_nome_particao(particao_i["nome_particao"]) in nomes_particoes):
            numeros_particoes.append(i)

    return numeros_particoes

def encuntar_nome_particao(nome):
    return os.path.splitext(os.path.split(nome)[1])[0]

def calcular_matriz_cr(particoes, func_comparacao):
    '''
    Gera matriz (NxN) contendo os CRs de todas as combinações possíveis entre as partições base

    :param particoes: Partições base
    '''
    matriz_cr = np.empty((len(particoes), len(particoes)))

    indices = particoes.index.tolist()
    for i in range(0, len(indices)):
        for j in range(i, len(indices)):
            matriz_cr[i][j] = func_comparacao(particoes.loc[i]["particao"]["label"],
                                                  particoes.loc[j]["particao"]["label"])

    rotulos = range(len(particoes))

    return pd.DataFrame(simetrizar_matriz(matriz_cr), index=rotulos, columns=rotulos)

def calcular_ari(p1, p2):
    return adjusted_rand_score(p1, p2)

def simetrizar_matriz(m):
    '''
    Copia o triângulo superior da matriz para o triângulo inferior

    src: https://stackoverflow.com/questions/17527693/

    :param m: Matriz a ser simetrizada
    '''
    inds = np.triu_indices_from(m, k=1) #k=1 é para ignorar a diagonal
    m[(inds[1], inds[0])] = m[inds]
    return m
