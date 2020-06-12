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

#-----------------------------------------------------------------------
## profiling del tiempo de ejecución de las dos implementaciones de la entropía. Se crean los datos, se activa el profiler, se realizan las ejecuciones, se desactiva el profiler.
# @param n número de muestras a considerar.
# @param d dimensión de las muestras.
# @param k número de vecinos más cercanos con los que calcular la entropía.
# @param reps número de veces que ejecutar las funciones.
# @return objeto de la clase cProfile.Profile con la información de las medidas realizadas.
def prof_ent(n, d, k = 3, reps = 10):
    # Generamos los datos de prueba
    dat = data.Data('normal', n, d)
    X = dat.X
    
    # Activamos el profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Ejecutamos ambos estimadores reps veces
    for i in range(0, reps):
         mi.entropy(X, k)
         ee.entropy(X, k)
    
    # Desactivamos el profiler
    pr.disable()
    return pr
    
#-----------------------------------------------------------------------
## profiling del tiempo de ejecución de las dos implementaciones de la información mutua. Se crean los datos, se activa el profiler, se realizan las ejecuciones, se desactiva el profiler.
# @param n número de muestras a considerar.
# @param d dimensión de las muestras.
# @param k número de vecinos más cercanos con los que calcular la entropía.
# @param reps número de veces que ejecutar las funciones.
# @return objeto de la clase cProfile.Profile con la información de las medidas realizadas
def prof_mi(n, d, k = 3, reps = 10):
    # Generamos la muestra 
    dat = data.Data('normal', n, 2*d)
    X = dat.X[:, 0:d]
    Y = dat.X[:, d:]
    
    # Activamos el profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Ejecutamos ambos estimadores reps veces
    for i in range(0, reps):
         mi.mutual_information((X, Y), k)
         ee.mi(X, Y, k=k)
    
    # Desactivamos el profiler
    pr.disable()
    return pr

## Almacena el tiempo de ejecución de cada función en la posición adecuada del dataframe correspondiente.
# @param p objeto de la clase pstats.Stats con las estadísticas sobre las ejecuciones.
# @param d dimensión con la que se realizó la medición.
# @param n número de muestras con la que se realizó la medición.
# @param df_mi dataframe en el que guardar la información relativa a la implementación mutual_info.py.
# @param df_ee dataframe en el que almacenar la información sobre la implementación entropy_estimators.py.
# @param list contiene las funciones de las que queremos obtener información.
def get_stats(p, d, n, df_mi, df_ee, list):
     # Para las funciones de la lista
     for f in list:
          # cc: número de llamadas excluyendo las recursivas
          # nc: número de llamadas
          # tt: tiempo total en la llamada (excluyendo el tiempo en llamadas a subfunciones)
          # ct: tiempo acumulado pasado en la función y todas sus subfunciones
          # callers: lista de las funciones que llamaron a esta función
          cc, nc, tt, ct, callers = p.stats[f]

          # Si la función es del archivo entropy_estimators.py, almacenamos la información en el df correspondiente
          if(f[0] == 'entropy_estimators.py'):
               # El valor que nos interesa es el tiempo pasado en la función (y a las que esta llama) entre el total de ejecuciones de la función
               df_ee.loc[d][n] = ct / nc

          # Análogamente, para mutual_info.py
          if(f[0] == 'mutual_info.py'):
               df_mi.loc[d][n] = ct / nc

## Obtiene los tiempos de ejecución de las dos implementaciones de la entropía y de las dos implementaciones de la información mutua. Imprime por pantalla los tiempos de ejecución y los guarda en archivos en la carpeta ./res.
# @param ds dimensiones que probar.
# @param ns tamañaos de muestra que probar.
# @param reps número de veces que repetir las ejecuciones.
# @param save si es True se almacenan los resultados en archivos.
def compare_estimators(ds, ns, reps = 5, save = False):
     # DataFrames para almacenar estimaciones
     mi_ent = pd.DataFrame(index = ds, columns = ns)
     ee_ent = pd.DataFrame(index = ds, columns = ns)
     mi_mi = pd.DataFrame(index = ds, columns = ns)
     ee_mi = pd.DataFrame(index = ds, columns = ns)
     
     # Variables necesarias para recuperar las mediciones con get_stats
     # Se obtuvieron mediante:
     # width, list = p.sort_stats(SortKey.CUMULATIVE).get_print_list('entropy')
     # p.print_line(list)
     # donde p es un objeto de pstats, entre la lista devuelta se seleccionaron las líneas correspondientes a las funciones a estudiar
     f_ent = [('entropy_estimators.py', 17, 'entropy'),
              ('mutual_info.py', 50, 'entropy')]
     f_mi = [('mutual_info.py', 91, 'mutual_information'),
             ('entropy_estimators.py', 61, 'mi')]
     
     # Realizaremos las medidas para todos los valores de d y n
     for d in ds:
         for n in ns:
             # Medimos tiempo de cálculos de entropía	
             pr = prof_ent(n, d, reps = reps)
             p_ent = pstats.Stats(pr).strip_dirs()
             get_stats(p_ent, d, n, mi_ent, ee_ent, f_ent)
          
             # Medimos tiempo de ejecución de información mutua
             pr = prof_mi(n, d, reps = reps)
             p_im = pstats.Stats(pr).strip_dirs()
             get_stats(p_im, d, n, mi_mi, ee_mi, f_mi)

         # Imprimimos en un archivo por cada función los resultados hasta el momento
         if (save):
             print(mi_ent, file=open('./res/mi_ent.txt', 'w'))
             print(ee_ent, file=open('./res/ee_ent.txt', 'w'))
             print(mi_mi, file=open('./res/mi_mi.txt', 'w'))
             print(ee_mi, file=open('./res/ee_mi.txt', 'w'))
        
    
     print("Entropía - mutual_info.py\n", mi_ent, "\n")
     print("Entropía - entropy_estimators.py\n", ee_ent, "\n")
     print("Información mutua - mutual_info.py\n", mi_mi, "\n")
     print("Información mutua - entropy_estimators.py\n", ee_mi, "\n")

     
## Imprime las 10 funciones que más tiempo consumen en la función que calcule la entropía.
# @param func función con la que calcular la entropía.
# @param n tamaño de muestra.
# @param d dimensión de los datos.
# @param reps número de veces que repetir la medición.
# @param k número de vecinos más cercanos con los que estimar la entropía.
def func_stats(func, reps, *args):
    # Generamos los datos de prueba
    
    # Activamos el profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Ejecutamos ambos estimadores reps veces
    for i in range(0, reps):
         func(*args)
         
    # Desactivamos el profiler
    pr.disable()
    
    p = pstats.Stats(pr).strip_dirs()
    p.sort_stats(SortKey.TIME).print_stats(10) # Imprime las 10 funciones que más tiempo consumen
    p.print_callers(10)

    
def main():
     # Valores de n y d para los que realizar estimaciones
     ns = [1000, 30000, 100000]
     ds = [2] #, 10, 100]
     reps = 5
     #compare_estimators(ds, ns, reps)
     n = 100000
     d = 4
     #entropy_stats(mi.entropy, n, d)
     #entropy_stats(ee.entropy, n, d)

     dat = data.Data('normal', n, d)
     X = dat.X

     # Vemos qué "subfunciones" de la entropía requieren más tiempo
     func_stats(mi.entropy, 5, X, 3)
     func_stats(ee.entropy, 5, X, 3)

     # Vemos qué subfunciones de la información mutua gastan más tiempo
     func_stats(mi.mutual_information, 5, (X, X), 3)
     func_stats(ee.mi, 5, X, X, None, 3)
     
if __name__ == "__main__":
    main()
