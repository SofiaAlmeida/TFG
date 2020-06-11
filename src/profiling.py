'''
   Archivo utilizado para realizar pruebas para comparar los tiempor de ejecución de dos implementaciones de estimadores de la entropía y de la información mutua.
'''

import entropy_estimators as ee
import mutual_info as mi
import data
import cProfile
import pstats
from pstats import SortKey
import pandas as pd

#-----------------------------------------------------------------------
def prof_ent(n, d, k = 3, reps = 10):
    # Generamos los datos de prueba
    dat = data.Data('normal', n, d)
    X = dat.X
    
    # Activamos el profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Ejecutamos ambos estimadores m veces
    for i in range(0, reps):
         mi.entropy(X, k)
         ee.entropy(X, k)
    
    # Desactivamos el profiler
    pr.disable()
    return pr
    
#-----------------------------------------------------------------------
def prof_im(n, d, k = 3, reps = 10):
    # Generamos la muestra 
    dat = data.Data('normal', n, 2*d)
    C = dat.C
    X = dat.X[:, 0:d]
    Y = dat.X[:, d:]
    
    # Activamos el profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Ejecutamos ambos estimadores m veces
    for i in range(0, reps):
         mi.mutual_information((X, Y), k)
         MI_est2 = ee.mi(X, Y, k=k)
    
    # Desactivamos el profiler
    pr.disable()
    return pr    
    
def get_stats(p, d, n, df_mi, df_ee, list):
     for f in list:
          cc, nc, tt, ct, callers = p.stats[f]
          
          if(f[0] == 'entropy_estimators.py'):
               df_ee.loc[d][n] = ct / nc
               
          if(f[0] == 'mutual_info.py'):
               df_mi.loc[d][n] = ct / nc
     
def main():
     # Valores de n y d para los que realizar estimaciones
     ns = [1000, 30000, 100000]
     ds = [2, 10, 100]

     # DataFrames para almacenar estimaciones
     mi_ent = pd.DataFrame(index = ds, columns = ns)
     ee_ent = pd.DataFrame(index = ds, columns = ns)
     mi_mi = pd.DataFrame(index = ds, columns = ns)
     ee_mi = pd.DataFrame(index = ds, columns = ns)
     
     # Variables necesarias para recuperar las mediciones
     # Se obtuvieron mediante % REVIEW
     f_ent = [('entropy_estimators.py', 17, 'entropy'), ('mutual_info.py', 50, 'entropy')]
     f_mi = [('mutual_info.py', 91, 'mutual_information'), ('entropy_estimators.py', 61, 'mi')]

     # Número de repeticiones
     reps = 5
     
     # Realizaremos las medidadas para todos los valores de d y n
     for d in ds:
         for n in ns:
             # Medimos tiempo de cálculos de entropía	
             pr = prof_ent(n, d, reps = reps)
             p_ent = pstats.Stats(pr).strip_dirs()
             get_stats(p_ent, d, n, mi_ent, ee_ent, f_ent)
          
             # Medimos tiempo de ejecución de información mutua
             pr = prof_im(n, d, reps = reps)
             p_im = pstats.Stats(pr).strip_dirs()
             get_stats(p_im, d, n, mi_mi, ee_mi, f_mi)
             
         print(mi_ent, file=open('./res/mi_ent.txt', 'w'))
         print(ee_ent, file=open('./res/ee_ent.txt', 'w'))
         print(mi_mi, file=open('./res/mi_mi.txt', 'w'))
         print(ee_mi, file=open('./res/ee_mi.txt', 'w'))
        
    
     print(mi_ent)
     print(ee_ent)
     print(mi_mi)
     print(ee_mi)
     
if __name__ == "__main__":
    main()
