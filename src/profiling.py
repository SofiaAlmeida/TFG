'''
   Pruebas para comparar los tiempos de ejecución de dos implementaciones de estimadores de la entropía y de la información mutua.
'''

import entropy_estimators as ee
import mutual_info as mi
import data
import cProfile
import pstats
from pstats import SortKey
import pandas as pd
import numpy as np

#-----------------------------------------------------------------------
## profiling del tiempo de ejecución de una función concreta. Se activa el profiler, se ejecuta la función, se desactiva el profiler.
# @param func función cuyos tiempos de ejecución queremos medir.
# @param args parámetros que pasar a la función a ejecutar.
# @return objeto de la clase cProfile.Profile con la información de las medidas realizadas.
def profile(func, *args):    
    # Activamos el profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Ejecutamos el estimador correspondiente
    func(*args)
    
    # Desactivamos el profiler
    pr.disable()
    
    # Devolvemos el profiler
    return pr

## Obtiene el tiempo de ejecución de la función indicada.
# @param p objeto de la clase pstats.Stats con las estadísticas sobre las ejecuciones.
# @param f función cuyos tiempos queremos calcular.
# @return el tiempo de ejecución de la función acumulado, es decir, incluyendo el tiempo de las funciones a las que esta llame.
def get_cumtime_percall(p, f):
      # cc: número de llamadas excluyendo las recursivas
      # nc: número de llamadas
      # tt: tiempo total en la llamada (excluyendo el tiempo en llamadas a subfunciones)
      # ct: tiempo acumulado pasado en la función y todas sus subfunciones
      # callers: lista de las funciones que llamaron a esta función
      cc, nc, tt, ct, callers = p.stats[f]

      # El valor que nos interesa es el tiempo pasado en la función (y a las que esta llama) entre el total de ejecuciones de la función
      return ct / nc


## Obtiene los tiempos de ejecución de las dos implementaciones de la entropía y de las dos implementaciones de la información mutua. Imprime por pantalla los tiempos de ejecución y los guarda en archivos en la carpeta ./res.
# @param ds dimensiones que probar.
# @param ns tamañaos de muestra que probar.
# @param k valor de k utilizado en las estimaciones.
# @param reps número de veces que repetir las ejecuciones.
# @param save si es True se almacenan los resultados en archivos.
def compare_estimators(ds, ns, k = 3, reps = 5, save = False):
     # DataFrame - MultiIndex para almacenar mediciones
     # Creamos nombres de filas y columnas
     iterables = [['ent', 'mi'], ds]
     index  = pd.MultiIndex.from_product(iterables, names = ['func', 'ds'])
     iterables2 = [['mi', 'ee'], ns]
     cols = pd.MultiIndex.from_product(iterables2, names = ['implementación', 'ns'])
     # Creamos el DataFrame
     df = pd.DataFrame(index = index, columns = cols)
     
     # Variables necesarias para recuperar las mediciones con get_cumtime_percall
     # Se obtuvieron mediante:
     # width, list = p.sort_stats(SortKey.CUMULATIVE).get_print_list('entropy')
     # p.print_line(list)
     # donde p es un objeto de pstats, entre la lista devuelta se seleccionaron las líneas correspondientes a las funciones a estudiar
     f_ent_mi = ('mutual_info.py', 50, 'entropy')
     f_ent_ee = ('entropy_estimators.py', 17, 'entropy')
     
     f_mi_mi = ('mutual_info.py', 91, 'mutual_information')
     f_mi_ee = ('entropy_estimators.py', 61, 'mi')
     
     # Realizaremos las medidas para todos los valores de d y n
     for d in ds:
         for n in ns:
             # Vectores para almacenar los tiempos en las diferentes repeticiones
             times_mi_ent = np.array([])
             times_ee_ent = np.array([])
             times_mi_mi = np.array([])
             times_ee_mi = np.array([])
             
             # Tomamos reps medidas
             for i in range(0, reps):
                 # Generamos los datos de prueba
                 dat = data.Data('normal', n, d)
                 X = dat.X
                 dat2 = data.Data('normal', n, d)
                 Y = dat2.X
                 
                 # Medimos tiempo de cálculos de entropía	
                 # mutual_info.py
                 pr_mi_ent = profile(mi.entropy, X, k)
                 stats = pstats.Stats(pr_mi_ent).strip_dirs()
                 times_mi_ent = np.append(times_mi_ent, 
                                          get_cumtime_percall(stats, f_ent_mi))
                 # Almacenamos las estadísticas
                 stats.dump_stats('stats/mutual_info_ent_' + str(n) + '_' + str(d) + '_' + str(i))
                 
                 # entropy_estimators.py
                 pr_ee_ent = profile(ee.entropy, X, k)
                 stats = pstats.Stats(pr_ee_ent).strip_dirs()
                 times_ee_ent = np.append(times_ee_ent, 
                                          get_cumtime_percall(stats, f_ent_ee))
                                          
                 stats.dump_stats('stats/ee_ent' + str(n) + '_' + str(d) + '_' + str(i))
                 
                 # Medimos tiempos de cálculo de la información mutua
                 # mutual_info.py 
                 pr_mi_mi = profile(mi.mutual_information, (X, Y) , k)
                 stats = pstats.Stats(pr_mi_mi).strip_dirs()
                 times_mi_mi = np.append(times_mi_mi, 
                                          get_cumtime_percall(stats, f_mi_mi))
                 stats.dump_stats('stats/mutual_info_mi_' + str(n) + '_' + str(d) + '_' + str(i))
                                          
                 # entropy_estimators.py
                 pr_ee_mi = profile(ee.mi, X, Y, None, k)
                 stats = pstats.Stats(pr_ee_mi).strip_dirs()
                 times_ee_mi = np.append(times_ee_mi, 
                                          get_cumtime_percall(stats, f_mi_ee))
                                          
                 stats.dump_stats('stats/ee_mi' + str(n) + '_' + str(d) + '_' + str(i))
             
             # Almacenamos los datos
             df.loc[('ent', d), ('mi', n)] = times_mi_ent
             df.loc[('ent', d), ('ee', n)] = times_ee_ent
             
             df.loc[('mi', d), ('mi', n)] = times_mi_mi
             df.loc[('mi', d), ('ee', n)] = times_ee_mi


             # Imprimimos en un archivo por cada función los resultados hasta el momento
             if (save):
                 df.to_pickle("./stats/times.pkl")

    
def main():
     # Valores de n y d para los que realizar estimaciones
     ns = [1000, 30000, 100000]
     ds = [2, 10, 25]
     reps = 5
     k = 3
     compare_estimators(ds, ns, k, reps, save = True)
     
     
if __name__ == "__main__":
    main()
