#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import unittest

import numpy as np
import pandas as pd

from .context import Leitor
from .context import calcular_matriz_cr, calcular_ari, obter_numeros_particoes

def verifica_se_eh_simetrica(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

class TestMatrizCRIris(unittest.TestCase):

    def setUp(self):
        leitor = Leitor()
        caminho = os.path.join(os.path.dirname(__file__), "resources", "iris-particoes",
                "iris-particoes")
        self.particoes = leitor.ler_particoes(caminho)
        self.matriz_cr = calcular_matriz_cr(self.particoes, calcular_ari)

    def test_calcular_matriz_cr_deve_retornar_data_frame_do_pandas(self):
        self.assertIsInstance(self.matriz_cr, pd.DataFrame, "A matriz deve ser um Dataframe")
        self.assertEqual(self.matriz_cr.shape, (37, 37), "A matriz deve ser NxN")
        self.assertTrue(verifica_se_eh_simetrica(self.matriz_cr), "A matriz deve ser simétrica")

    def test_diagonal_da_matriz_deve_ser_composta_por_1s(self):
        self.assertTrue((np.diagonal(self.matriz_cr) == 1).all(),
                "As partições devem ser iguais a elas mesmas")

class TestMatrizCRFake(unittest.TestCase):

    def setUp(self):
        leitor = Leitor()
        caminho = os.path.join(os.path.dirname(__file__), "resources", "outras-particoes",
                "fake")
        self.particoes = leitor.ler_particoes(caminho)
        self.matriz_cr = calcular_matriz_cr(self.particoes, calcular_ari)

    def test_calcular_matriz_cr_deve_retornar_data_frame_do_pandas(self):
        self.assertIsInstance(self.matriz_cr, pd.DataFrame, "A matriz deve ser um Dataframe")
        self.assertEqual(self.matriz_cr.shape, (3, 3), "A matriz deve ser NxN")
        self.assertTrue(verifica_se_eh_simetrica(self.matriz_cr), "A matriz deve ser simétrica")

    def test_ignorando_a_diagonal_deve_haver_duas_particoes_iguais(self):
        self.assertEqual(np.count_nonzero(self.matriz_cr == 1), 5, "Deve haver duas partições iguais")

if __name__ == "__main__":
    unittest.main()
