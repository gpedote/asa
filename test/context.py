import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../leitor")))

from asa.asa import ASA
from leitor.leitor_particoes import Leitor
from util.util import calcular_matriz_cr, calcular_ari, obter_numeros_particoes
