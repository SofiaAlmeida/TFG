
'''
   Archivo utilizado para realizar pruebas para comparar dos implementaciones de estimadores de la entropía y de la información mutua.
'''

import entropy_estimators as ee
import mutual_info as mi
import data
import cProfile
import pstats
from pstats import SortKey


import timeit

#-----------------------------------------------------------------------
def prof(n, d, k = 3):
    dat = data.Data('normal', n, d)
    X = dat.X
    
    cProfile.runctx('mi.entropy(X, k)', globals(), locals(), 'stats-mi')
    cProfile.runctx('ee.entropy(X, k)', globals(), locals(), 'stats-ee')
    

def main():  
     prof(50000, 2)
     p = pstats.Stats('stats-mi', 'stats-ee')
     p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats('entropy') # Imprime por pantalla las funciones que contengan 'entropy' en su nombre, ordenadas por tiempo acumulado

if __name__ == "__main__":
    main()
