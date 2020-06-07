'''
   Archivo utilizado para realizar pruebas para comparar los tiempor de ejecución de dos implementaciones de estimadores de la entropía y de la información mutua.
'''

import entropy_estimators as ee
import mutual_info as mi
import data
import cProfile
import pstats
from pstats import SortKey

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
    
def get_stats(p, v, dic_mi, dic_ee, list):
     for f in list:
          cc, nc, tt, ct, callers = p.stats[f]
          
          if(f[0] == 'entropy_estimators.py'):
               dic_ee[v] = ct / nc
               
          if(f[0] == 'mutual_info.py'):
               dic_mi[v] = ct / nc
     
def main():  	
     d = 2
     ns = [5000, 50000, 100000, 150000, 200000]
     n_mi_ent = {}
     n_ee_ent ={}
     f_ent = [('entropy_estimators.py', 17, 'entropy'), ('mutual_info.py', 50, 'entropy')]
     
     n_mi_mi = {}
     n_ee_mi ={}
     f_mi = [('mutual_info.py', 91, 'mutual_information'), ('entropy_estimators.py', 61, 'mi')]
     
     # Fijamos un valor d y medimos los tiempos de ejecución de la entropía y la información mutua para distintos valores de n
     for n in ns:		
          # Medimos tiempo de cálculos de entropía	
          pr = prof_ent(n, d, reps = 20)
          p_ent = pstats.Stats(pr).strip_dirs()
          get_stats(p_ent, n, n_mi_ent, n_ee_ent, f_ent)
          
          # Medimos tiempo de ejecución de información mutua
          pr = prof_im(n, d, reps = 10)
          p_im = pstats.Stats(pr).strip_dirs()
          get_stats(p_im, n, n_mi_mi, n_ee_mi, f_mi)
          
     print(n_mi_ent)
     print(n_ee_ent)
     print(n_mi_mi)
     print(n_ee_mi)
     
     ds = range(2, 6)
     n = 50000
     reps = 10
     d_mi_ent = {}
     d_ee_ent ={}
     d_mi_mi = {}
     d_ee_mi ={}
     
     # Fijamos el valor de n y medimos los tiempos de ejecución de la entropía y la im para distintos valores de d
     for d in ds:
          # Medimos tiempo de cálculos de entropía	
          pr = prof_ent(n, d, reps = reps)
          p_ent = pstats.Stats(pr).strip_dirs()
          get_stats(p_ent, d, d_mi_ent, d_ee_ent, f_ent)
          
          # Medimos tiempo de ejecución de información mutua
          pr = prof_im(n, d, reps = reps)
          p_im = pstats.Stats(pr).strip_dirs()
          get_stats(p_im, d, d_mi_mi, d_ee_mi, f_mi)
          
     print(d_mi_ent)
     print(d_ee_ent)
     print(d_mi_mi)
     print(d_ee_mi)
     
if __name__ == "__main__":
    main()
