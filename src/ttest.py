from scipy import stats
import random
import numpy as np
import pandas as pd
import entropy_estimators as ee
import mutual_info as mi
import data


def ttest(N, ds, ns, k, save):
     # DataFrame - MultiIndex para almacenar mediciones
     # Creamos nombres de filas y columnas
     iterables = [ds, ['p', 't']]
     index  = pd.MultiIndex.from_product(iterables, names = ['ds', 'test-res'])
     iterables2 = [['ent', 'mi'], ns]
     cols = pd.MultiIndex.from_product(iterables2, names = ['función', 'ns'])
     # Creamos el DataFrame
     df = pd.DataFrame(index = index, columns = cols)    
     
     for d in ds:
         for n in ns:
             # ENTROPÍA
             # Generamos los datos
             X = [data.Data('random', n, d).X for i in range(0, N)]
             
             # Calculamos la entropía
             mi_ent = np.array([mi.entropy(x, k = k) for x in X])  
             ee_ent = np.array([ee.entropy(x, k = k) for x in X])
    
             # Calculamos el ttest
             t, p = stats.ttest_rel(mi_ent, ee_ent)
             
             # Almacenamos los datos
             df.loc[(d, 'p'), ('ent', n)] = p 
             df.loc[(d, 't'), ('ent', n)] = t
             
             # INFORMACIÓN MUTUA
             Y = [data.Data('random', n, d).X for i in range(0, N)]    
             mi_mi = np.array([mi.mutual_information((x, y), k = k) for x, y in zip(X, Y)])
             ee_mi = np.array([ee.mi(x, y, k = k) for x, y in zip(X, Y)])
             
             # Calculamos el ttest
             t, p = stats.ttest_rel(mi_mi, ee_mi)
             
             # Almacenamos los datos
             df.loc[(d, 'p'), ('mi', n)] = p
             df.loc[(d, 't'), ('mi', n)] = t
             
             # Imprimimos en un archivo por cada función los resultados hasta el momento
             if (save):
                 df.to_pickle("./stats/ttest.pkl")

def test_normality(N, ds, ns, k):
     for d in ds:
         for n in ns:
             print("ENTROPÍA")
             # Generamos los datos
             X = [data.Data('random', n, d).X for i in range(0, N)]
             
             # Calculamos la entropía
             mi_ent = np.array([mi.entropy(x, k = k) for x in X])  
             ee_ent = np.array([ee.entropy(x, k = k) for x in X])

             D = [X1 - X2 for X1, X2 in zip(mi_ent, ee_ent)]
             
             k2, p = stats.normaltest(D)
             
             print ("d: ", d, ", n: ", n)
             if p <= 0.05:  # Hipótesis nula: D proviene de una distribución normal
                 print("Podemos rechazar la hipótesis nula")
             else:
                 print("No podemos rechazar la hipótesis nula")
                 
             print("INFORMACIÓN MUTUA")    
             Y = [data.Data('random', n, d).X for i in range(0, N)]    
             
             mi_mi = np.array([mi.mutual_information((x, y), k = k) for x, y in zip(X, Y)])
             ee_mi = np.array([ee.mi(x, y, k = k) for x, y in zip(X, Y)])
             
             D = [X1 - X2 for X1, X2 in zip(mi_mi, ee_mi)]
             
             k2, p = stats.normaltest(D)
             
             print ("d: ", d, ", n: ", n)
             if p <= 0.05:  # Hipótesis nula: D proviene de una distribución normal
                 print("Podemos rechazar la hipótesis nula")
             else:
                 print("No podemos rechazar la hipótesis nula")
                 
#-----------------------------------------------------------------------
def main():
    np.random.seed(0)
    N = 30
    n = 30000
    k = 3
    ds = [2, 10, 25]
    ns = [1000, 30000, 100000]
    
    test_normality(N, ds, ns, k)
    #ttest(N, ds, ns, k, save = True)
    
    
if __name__ == "__main__":
    main()