#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import unittest

import numpy as np
import pandas as pd

from .context import ASA
from .context import Leitor
from .context import calcular_matriz_cr, calcular_ari, obter_numeros_particoes

def embaralhar_particoes(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df

# Armazena a solução correta para cada entrada de quantidade máxima de partições iguais a serem
# removidas
SOLUCOES_CORRETAS_MAXIMO_PARTS = {
    0: ["iris-SL-E-k6", "iris-SNN-E-NN60-t0.8-n0-m0.4", "iris-SNN-E-NN60-t0.5-n0-m0.8",
        "iris-KM-E-k6-Run22", "iris-AL-E-k4", "iris-KM-E-k2-Run2", "iris-SL-E-k5",
        "iris-CoL-E-k4", "iris-SNN-E-NN45-t0.8-n0-m0.4", "iris-KM-E-k3-Run6", "iris-CoL-E-k3",
        "iris-SL-E-k4", "iris-CoL-E-k6", "iris-CeL-E-k3", "iris-KM-E-k5-Run1", "iris-CeL-E-k6",
        "iris-AL-E-k5", "iris-KM-E-k4-Run22", "iris-AL-E-k3", "iris-SNN-E-NN30-t0.8-n0-m0.6",
        "iris-CeL-E-k5", "iris-SL-E-k3", "iris-CoL-E-k5", "iris-CoL-E-k2", "iris-AL-E-k6",
        "iris-CeL-E-k4"],
    2: ["iris-SNN-E-NN60-t0.8-n0-m0.4", "iris-SNN-E-NN60-t0.5-n0-m0.8", "iris-CoL-E-k4",
        "iris-CeL-E-k3", "iris-KM-E-k4-Run22", "iris-CoL-E-k6", "iris-KM-E-k6-Run22",
        "iris-CoL-E-k2"],
    3: ["iris-SNN-E-NN60-t0.5-n0-m0.8", "iris-CoL-E-k4", "iris-CeL-E-k3", "iris-KM-E-k4-Run22",
        "iris-CoL-E-k6", "iris-KM-E-k6-Run22", "iris-CoL-E-k2"]
}

class TestASA(unittest.TestCase):

    def setUp(self):
        leitor = Leitor()
        caminho = os.path.join(os.path.dirname(__file__), "resources", "iris-particoes",
                "iris-particoes")
        self.particoes = leitor.ler_particoes(caminho)
        self.asa = ASA(self.particoes)
        self.matriz_cr = calcular_matriz_cr(self.particoes, calcular_ari)

    def test_verifica_corretude_com_zero_particoes_iguais(self):
        self.__verificar_se_resultado_esta_correto(self.asa.asa(0),
                SOLUCOES_CORRETAS_MAXIMO_PARTS[0])

    def test_verifica_corretude_com_duas_particoes_iguais(self):
        self.__verificar_se_resultado_esta_correto(self.asa.asa(2),
                SOLUCOES_CORRETAS_MAXIMO_PARTS[2])

    def test_verifica_corretude_com_tres_particoes_iguais(self):
        self.__verificar_se_resultado_esta_correto(self.asa.asa(3),
                SOLUCOES_CORRETAS_MAXIMO_PARTS[3])

    def test_garante_que_resultado_nao_eh_dependente_da_ordem_das_particoes_de_entrada(self):
        embaralhar_particoes(self.particoes)

        # É necessário reinicializar depois do embaralhamento
        self.asa = ASA(self.particoes)
        self.__verificar_se_resultado_esta_correto(self.asa.asa(2),
                SOLUCOES_CORRETAS_MAXIMO_PARTS[2])

    def __verificar_se_resultado_esta_correto(self, resultado, resultado_correto):
        indices_resultado_correto = obter_numeros_particoes(self.particoes, resultado_correto)

        for e in resultado:
            valores = []
            for j in indices_resultado_correto:
                valores.append(self.matriz_cr.loc[e][j])
            self.assertTrue(1 in valores, "Deve haver pelo menos uma partição"
                    + "igual incluída no resultado atual que seja identica ao resultado anterior")

class TestParticoesDuplicadas(unittest.TestCase):

    def setUp(self):
        leitor = Leitor()
        caminho = os.path.join(os.path.dirname(__file__), "resources", "outras-particoes",
                "duplicadas")
        self.particoes = leitor.ler_particoes(caminho)
        self.asa = ASA(self.particoes)

    def test_garante_que_esta_removendo_as_duplicadas(self):
        self.assertEqual(len(self.asa.asa(2)), 1, "Deve remover as duplicadas")

    def test_garante_que_esta_removendo_as_triplicadas(self):
        self.assertEqual(len(self.asa.asa(3)), 1, "Deve remover as triplicadas")

if __name__ == "__main__":
    # import sys;sys.argv = [""]
    unittest.main()
