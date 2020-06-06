'''
   Archivo utilizado para realizar pruebas para comparar los tiempor de ejecución de dos implementaciones de estimadores de la entropía y de la información mutua.
'''

import entropy_estimators as ee
import mutual_info as mi
import data
import cProfile
import pstats
from pstats import SortKey

import timeit

#-----------------------------------------------------------------------
def prof(n, d, k = 3, reps = 10):
    dat = data.Data('normal', n, d)
    X = dat.X
    
    pr = cProfile.Profile()
    pr.enable()
    
    for i in range(0, reps):
         mi.entropy(X, k)
         ee.entropy(X, k)
    
    pr.disable()
    return pr

def main():  	
     ns = [5000, 50000, 100000, 150000, 200000]
     for n in ns:	
          print("Número de muestras: ", n)			
          pr = prof(n, 2)
          p = pstats.Stats(pr)
          p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats('entropy') # Imprime por pantalla las funciones que contengan 'entropy' en su nombre, ordenadas por tiempo acumulado
     
if __name__ == "__main__":
    main()
