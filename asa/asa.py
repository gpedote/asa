#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from context import calcular_matriz_cr, calcular_ari

class ASA(object):

    LIMIAR_RAZAO_ASA = 0.12
    LIMIAR_INICIAL_ASA = 0.9
    LIMIAR_FINAL_ASA = 0.1
    RAZAO_INICIAL_ASA = 1.0

    def __init__(self, particoes, exibir_dados_processamento=False):
        self.particoes = particoes
        self.exibir_dados_processamento = exibir_dados_processamento

    def asa(self, max_parts_identicas):
        matriz_cr = calcular_matriz_cr(self.particoes, calcular_ari)
        matriz_cr_original = matriz_cr.copy()
        (parts_evidentes, parts_duplicadas) = self.__obter_particoes_evidentes_e_duplicadas(
                matriz_cr, max_parts_identicas)
        parts_duplicadas.extend(parts_evidentes.index.tolist())
        matriz_cr = self.__obter_matriz_cr_sem_particoes_duplicadas(matriz_cr, parts_duplicadas)

        # Calcula CR médio das partições restantes
        medias_cr = matriz_cr.mean(axis=0)

        # Armazena dados do processamento do ASA
        colunas_processamento = ["limite_cr", "numero_de_solucoes", "razao", "razao_anterior",
                "dif_razoes"]
        dados_processamento = pd.DataFrame([], columns=colunas_processamento)

        # Calcula número inicial de partições
        numero_de_solucoes = len(matriz_cr) + len(parts_evidentes)

        limite_cr = limite_cr_anterior = self.LIMIAR_INICIAL_ASA
        razao = razao_anterior = self.RAZAO_INICIAL_ASA

        matriz_cr_atual = matriz_cr.copy()
        media_cr_atual = medias_cr.copy()

        matriz_cr_anterior = pd.DataFrame([], columns=matriz_cr.keys())
        media_cr_anterior = pd.Series([])

        solucao_encontrada = False
        while (not solucao_encontrada):
            razao_anterior = razao
            matriz_cr_anterior = matriz_cr_atual.copy()
            media_cr_anterior = media_cr_atual.copy()

            # Limpa variáveis
            matriz_cr_atual = pd.DataFrame([], columns=matriz_cr.keys())
            media_cr_atual = pd.Series([])

            # Remove todas as partições onde CR(pI, pJ) >= limite_cr com pI ∈ R e pJ ∈ C
            particoes_a_serem_removidas = []
            for i in parts_evidentes.index.tolist():
                for j in matriz_cr.index.tolist():
                    if (self.__eh_maior_ou_igual_float(matriz_cr_original[i][j], limite_cr)):
                        particoes_a_serem_removidas.append(j)

            # Remove duplicações
            particoes_a_serem_removidas = np.unique(particoes_a_serem_removidas)

            # Remove partições onde CR >= limite_cr do eixo 0
            matriz_cr = matriz_cr.drop(particoes_a_serem_removidas, axis=0)
            medias_cr = medias_cr.drop(particoes_a_serem_removidas, axis=0)

            # Passos 7d e 7e
            while (len(matriz_cr) > 0):
                part_com_maior_cr_medio = np.argmax(medias_cr)

                # Armazena na solução atual a partição com o maior CR médio
                matriz_cr_atual.loc[part_com_maior_cr_medio] = matriz_cr.loc[part_com_maior_cr_medio]
                media_cr_atual.loc[part_com_maior_cr_medio] = medias_cr.loc[part_com_maior_cr_medio]

                # Remove do C  a partição com o maior CR médio
                matriz_cr = matriz_cr.drop(part_com_maior_cr_medio, axis=0)
                medias_cr = medias_cr.drop(part_com_maior_cr_medio, axis=0)

                # Excluí partições onde CR(part_com_maior_cr_medio, J) maiores que limite_cr
                particoes_a_serem_removidas = []
                for j in matriz_cr.index.tolist():
                    if (self.__eh_maior_ou_igual_float(
                            matriz_cr_original[part_com_maior_cr_medio][j], limite_cr)):
                        particoes_a_serem_removidas.append(j)

                # Remove duplicações
                particoes_a_serem_removidas = np.unique(particoes_a_serem_removidas)

                # Remove partições onde CR(part_com_maior_cr_medio, J) maiores que limite_cr
                matriz_cr = matriz_cr.drop(particoes_a_serem_removidas, axis=0)
                medias_cr = medias_cr.drop(particoes_a_serem_removidas, axis=0)

            # Atualiza C
            matriz_cr = matriz_cr_atual.copy()
            medias_cr = media_cr_atual.copy()

            # Calcula diferença em % entre o número partições da solução atual e a qtd de
            # soluções inicial
            razao = len(matriz_cr_atual) / float(numero_de_solucoes)

            # Coleta dados
            dados_processamento = dados_processamento.append(pd.DataFrame([[
                    limite_cr, len(matriz_cr_atual),razao, razao_anterior, razao_anterior - razao
                    ]], columns=colunas_processamento))
            if (self.__verificar_se_ultrapassou_o_limiar_razao(razao, razao_anterior)
                    and self.__eh_maior_ou_igual_float(limite_cr, self.LIMIAR_FINAL_ASA)):
                # Atualiza limite_cr, pois caso contrário, não ficaria claro se o limite_cr_anterior
                # ou o limite_cr é o limite pelo qual as partições da solução foram escolhidas
                limite_cr_anterior = limite_cr
                limite_cr -= 0.1
            else:
                solucao_encontrada = True

        if (self.exibir_dados_processamento):
            print dados_processamento

        return self.__obter_resultado_final(parts_evidentes, limite_cr, matriz_cr_anterior)

    def __eh_maior_ou_igual_float(self, a, b):
        return a > b or np.isclose(a, b)

    def __verificar_se_ultrapassou_o_limiar_razao(self, razao, razao_anterior):
        return (razao_anterior - razao > self.LIMIAR_RAZAO_ASA)

    def __obter_resultado_final(self, parts_evidentes, limite_cr, matriz_cr_anterior):
        # Solução final, passo 8 do artigo
        solucao_final = parts_evidentes.index.tolist()
        solucao_final.extend(matriz_cr_anterior.index.tolist())

        return solucao_final

    def __obter_matriz_cr_sem_particoes_duplicadas(self, matriz_cr, parts_duplicadas):
        # Remove duplicações
        parts_a_serem_removidas = np.unique(parts_duplicadas)

        # Remove partições duplicadas do eixo 0 e 1
        matriz_cr = matriz_cr.drop(parts_a_serem_removidas, axis=0)
        matriz_cr = matriz_cr.drop(parts_a_serem_removidas, axis=1)
        return matriz_cr

    def __obter_particoes_evidentes_e_duplicadas(self, matriz_cr, max_parts_identicas=2):
        '''
        Obtém particoes mais evidentes e particoes duplicadas

        :param max_parts_identicas: Número máximo que uma partição pode se repetir,
        para ser considerada uma partição evidente
        '''
        parts_evidentes = pd.DataFrame([], columns=self.particoes.columns)
        parts_duplicadas = []

        for i, crs in matriz_cr.iterrows():
            if i not in parts_duplicadas:
                parts_identicas = np.flatnonzero(np.isclose(crs, 1)).tolist()
                parts_identicas.remove(i)

                if (len(parts_identicas) >= max_parts_identicas):
                    parts_evidentes.loc[i] = self.particoes.loc[i]
                    parts_duplicadas.extend(parts_identicas)

        return (parts_evidentes, parts_duplicadas)
