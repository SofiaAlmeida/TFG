
'''
   Archivo utilizado para realizar pruebas para comparar dos implementaciones de estimadores de la entropía y de la información mutua.
'''

import entropy_estimators as ee
import mutual_info as mi
import pruebas
import numpy as np
import cProfile
import pstats
from pstats import SortKey
import sys

#-----------------------------------------------------------------------
cProfile.run('pruebas.rd_ent(50000, 2, 2, 3)', 'restats')

p = pstats.Stats('restats')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10) # Imprime por pantalla las 10 funciones que más tiempo consumen

# Imprime en el archivo stats.txt
sys.stdout = open('stats.txt', 'w')
p = pstats.Stats('restats')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
