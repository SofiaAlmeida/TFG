
'''
   Archivo utilizado para realizar pruebas para comparar dos implementaciones de estimadores de la entropía y de la información mutua.

    Copyright (C) 2020  Sofía Almeida

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''

import entropy_estimators as ee
import mutual_info as mi
import data

import random
from numpy import pi
from scipy.special import gamma, psi
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#----------------------------------------------------------------------
## Calcula la entropía de 2 variables aleatorias con ambos estimadores.
# Estima la entropía de las variables x e y y la imprime por pantalla.
# @param k número de vecinos a considerar al llamar a los estimadores.
# @param x primera variable aleatoria.
# @param y segunda variable aleatoria.
def ent(k, x, y):
    print("Entropía de X:")
    print("Estimador 1: ", mi.entropy(x, k = k))
    print("Estimador 2: ", ee.entropy(x))

    print("Entropía de Y:")
    print("Estimador 1: ", mi.entropy(y, k = k))
    print("Estimador 2: ", ee.entropy(y))

#---------------------------------------------------------------------
## Adaptación de mutual_info.test_entropy para que se compare también entropy_estimators.entropy con una normal.
# @param dat objeto de la clase Data generado mediante una distribución normal.
# @param k número de vecinos a utilizar en los estimadores.
def test_entropy(dat, k = 3):
    C = dat.C
    X = dat.X

    H_th = mi.entropy_gaussian(C) / np.log(2)
    H_est = mi.entropy(X, k) 
    H_est2 = ee.entropy(X, k)

    print("Entropía gaussiana:")
    print("Teórica: ", H_th)
    print("Estimador 1: ", H_est)
    print("Estimador 2: ", H_est2)

    # La entropía estimada debe ser menor que la teórica pero no por mucho
    np.testing.assert_array_less(H_est, H_th)
    np.testing.assert_array_less(H_est2, H_th)
    np.testing.assert_array_less(.9*H_th, H_est)
    np.testing.assert_array_less(.9*H_th, H_est2)
    

#---------------------------------------------------------------------
## Modificación de pruebas.test_entropy, devuelve la diferencia entre la entropía teórica y las estimadas de variable aleatoria con distribución normal.
# @param dat objeto de la clase Data que contiene una variable aleatoria con distribución normal.
# @param k número de vecinos más cercanos a considerar para realizar las estimacioines.
# @param show si vale True imprimirá por pantalla los resultados, si vale False, no.
# @return Diferencia entre la estimación teórica y la estimación realizada por el primer estimador, diferencia entre la estimación teórica y la estimación realizada por el segundo estimador.
def err_entropy(dat, k = 3, show = False):
    C = dat.C
    X = dat.X
    
    # Calculamos las entropías
    H_th = mi.entropy_gaussian(C) / np.log(2)
    H_est = mi.entropy(X, k)
    H_est2 = ee.entropy(X, k)

    if(show):
        print("Entropía gaussiana:")
        print("Teórica: ", H_th)
        print("Estimador 1: ", H_est)
        print("Estimador 2: ", H_est2)

    # Calculamos la diferencia entre la entropía teórica y la estimada
    dif1 = abs(H_th - H_est)
    dif2 = abs(H_th - H_est2)
    
    return dif1, dif2

#---------------------------------------------------------------------
## Calcula la diferencia entre la entropía teórica y las estimadas
# para variables aleatorias con distribución normal con dimensiones desde min_d hasta max_d. El error se calcula como la media de los errores en las repeticiones de la ejecución para cada dimensión. En cada repetición se calcula el error utilizando pruebas.err_entropy. Imprime por pantalla los errores en cada dimensión y al final muestra una gráfica de los mismos.
# @param reps número de veces que se calculará la entropía en cada dimensión.
# @param n número de muestras de la variable aleatoria a tomar.
# @param min_d dimensión inicial.
# @param max_d dimensión final.
# @param k número de vecinos más cercano con el que realizar las estimaciones.
def exp_err_ent(reps = 100, n = 50000, min_d = 2, max_d = 9, k = 3):    
    err1 = np.array([])
    err2 = np.array([])
    
    for d in range(min_d, max_d + 1):
        dif1 = np.array([])
        dif2 = np.array([])
        for i in range(0, reps):
            # Generamos datos
            dat = data.Data('normal', n, d)
            # Calculamos el error
            d1, d2 = err_entropy(dat, k)
            # Añadimos el error en esta repetición
            dif1 = np.append(dif1, d1)
            dif2 = np.append(dif2, d2)
        # Calculamos la media de los errores para esta dimensión
        e1 = np.mean(dif1)
        e2 = np.mean(dif2)
        # Los añadimos a la lista de errores
        err1 = np.append(err1, e1)
        err2 = np.append(err2, e2)

        print("d = ", d)
        print("Error estimador 1: ", e1)
        print("Error estimador 2: ", e2)
 
    plot_err(err1, err2, min_d, max_d)


#---------------------------------------------------------------------
## Cálculo de entropía para un caso concreto.
# No va a dar resultados correctos porque es una variable discreta y los estimadores son para variables continuas.
def example_ent():
    x = np.random.choice([1.0,2.0,3.0,4.0], (1000, 1) , p=[0.5, 0.25, 0.125, 0.125])
    
    H_est = mi.entropy(x, k=5) 
    H_est2 = ee.entropy(x, k=5)
    
    print("Entropía x:")
    print("Teórica: ", 7.0/4.0)
    print("Estimador 1: ", H_est)
    print("Estimador 2: ", H_est2)

    y = np.random.choice([1.0,2.0,3.0,4.0], (1000, 1) , p=[0.25, 0.25, 0.25, 0.25])
    
    H_est = mi.entropy(y, k=5) 
    H_est2 = ee.entropy(y, k=5)
    
    print("Entropía y:")
    print("Teórica: ", 2.0)
    print("Estimador 1: ", H_est)
    print("Estimador 2: ", H_est2)

#---------------------------------------------------------------------
## Imprime el cálculo de la entropía de una variable aleatoria con distribución uniforme en el intervalor [a,b], teórica y estimada.
# @param dat dat objeto de la clase Data con distribución uniforme.
# @param k número de vecinos más cercanos con los que realizar las estimaciones.
def example_ent_unif(dat, k = 3):
    x = dat.X
    a = dat.a
    b = dat.b
    
    H_est = mi.entropy(x, k) 
    H_est2 = ee.entropy(x, k)

    print("Entropía x:")
    print("Teórica: ", np.log(b - a) / np.log(2))
    print("Estimador 1: ", H_est)
    print("Estimador 2: ", H_est2)

#---------------------------------------------------------------------
## Muestra la gráfica de dos vectores en un intervalo dado. Los vectores serán los errores cometidos por cada estimador para las dimensiones [min_d, max_d].
# @param x primer vector.
# @param y segundo vector.
# @param min_d extremo infreior del intervalo.
# @param max_d extremo superior del intervalo.
def plot_err(x, y, min_d, max_d):
    plt.plot(range(min_d, max_d + 1), x, label = "Error estimador 1")
    plt.plot(range(min_d, max_d + 1), y, label = "Error estimador 2")
    plt.legend(loc="upper left")
    plt.xlabel("d, Dimensión")
    plt.ylabel("bits")
    plt.show()   

#-----------------------------------------------------------------------
## Adaptación de mutual_information.test_mutual_information para incluir a entropy_estimators.mi. Calcula la información mutua de dos variables aleatorias de dimensión 2 con distribución normal. Imprime por pantalla la información mutua teórica y la estudiada por los 2 estimadores.
# @param dat objeto de la clase Data con distribución normal y dimensión 4.
# @param k número de vecinos más cercanos a utilizar para realizar la estimación.
def test_mutual_information(dat, k = 3):
    C = dat.C
    X = dat.X[:,0:2]
    Y = dat.X[:,2:4]
    
    MI_est = mi.mutual_information((X, Y), k = k)
    MI_est2 = ee.mi(X, Y, k = k)
    MI_th = (mi.entropy_gaussian(C[0, 0]) / np.log(2)
             + mi.entropy_gaussian(C[1, 1]) / np.log(2)
             + mi.entropy_gaussian(C[2, 2]) / np.log(2)
             + mi.entropy_gaussian(C[3, 3]) / np.log(2)
             - mi.entropy_gaussian(C) / np.log(2)
            )

    # Imprimimos los resultados
    print("IM gaussiana:")
    print("Teórica: ", MI_th)
    print("Estimador 1: ", MI_est)
    print("Estimador 2: ", MI_est2)
    print("Error 1: ", abs(MI_est - MI_th))
    print("Error 2: ", abs(MI_est2 - MI_th))

    np.testing.assert_array_less(MI_est, MI_th)
    np.testing.assert_array_less(MI_th, MI_est  + .3)
    
    np.testing.assert_array_less(MI_est2, MI_th)
    np.testing.assert_array_less(MI_th, MI_est2  + .3)

#-----------------------------------------------------------------------
## Modificación de test_mutual_information para que imprima la diferencia entre los estimadores teóricos y estimados, además, permite que las variables tengan la dimensión indicada.
# @param dat objeto de la clase Data con distribución normal y dimensión 2*d.
# @param k número de vecinos más cercanos a utilizar en los estimadores.
# @param show si True, se imprimen resultados por pantalla, si False, no.
# @return Diferencia entre la información mutua teórica y la estimada por la primera implementación, diferencia entre la información mutua teórica y la estimada por la segunda implementación.
def test_mutual_information_mod(dat, k = 3, show = False):
    d = int(dat.d / 2)
    C = dat.C
    X = dat.X[:, 0:d]
    Y = dat.X[:,d:]
    
    # Estimamos
    MI_est = mi.mutual_information((X, Y), k)
    MI_est2 = ee.mi(X, Y)

    # Información mutua teórica
    MI_th = 0
    for i in range(0, 2*d):
        MI_th += mi.entropy_gaussian(C[i, i]) / np.log(2)
  
    MI_th -=  mi.entropy_gaussian(C) / np.log(2)

    # Calculamos las diferencias
    dif1 = abs(MI_est - MI_th)
    dif2 = abs(MI_est2 - MI_th)

    if show:
        print("IM gaussiana:")
        print("Teórica: ", MI_th)
        print("Estimador 1: ", MI_est)
        print("Estimador 2: ", MI_est2)
        print("Error 1: ", dif1)
        print("Error 2: ", dif2)

    return dif1, dif2

#---------------------------------------------------------------------
## Calcularemos los errores obtenidos con ambos estimadores para dimensiones desde min_d hasta max_d, repitiendo reps veces el cálculo por dimensión y devolviendo la media de las diferencias con la entropía teórica. Imprime los errores por pantalla y muestra una gráfica con los vectores de errores.
# @param reps número de veces a repetir el cálculo por dimensión.
# @param n número de ejemplos de las variables a considerar.
# @param min_d dimensión inicial.
# @param max_d dimensión final.
# @param k número de vecinos más cercanos con los que estimar.
def exp_err_im(reps = 100, n = 50000, min_d = 2, max_d = 9, k = 3):    
    err1 = np.array([])
    err2 = np.array([])
    
    for d in range(min_d, max_d + 1):
        dif1 = np.array([])
        dif2 = np.array([])
        for i in range(0, reps):
            dat = data.Data('normal', n, 2*d)
            d1, d2 = test_mutual_information_mod(dat, k)
            dif1 = np.append(dif1, d1)
            dif2 = np.append(dif2, d2)

        e1 = np.mean(dif1)
        e2 = np.mean(dif2)
        err1 = np.append(err1, e1)
        err2 = np.append(err2, e2)

        print("d = ", d)
        print("Error estimador 1: ", e1)
        print("Error estimador 2: ", e2)

    plot_err(err1, err2, min_d, max_d)

    
#-----------------------------------------------------------------------
def main():
    np.random.seed(0)
    #-------------------------------
    # Calculamos la entropía de dos variables aleatorias
    N = 1000
    x = data.Data('random', N, 3).X
    y = data.Data('random', N, 4).X
    #ent(3, x, y)
    
    # Entropía de una variable aleatoria con distribución normal
    N = 50000
    d = 2
    dat = data.Data('normal', N, d)
    #test_entropy(dat, 3)
    #err_entropy(dat, 3, True)
    #err_entropy_mod(dat, 3)
    
    # # example_ent()
    # Calculamos entropía va con distribución uniforme
    dat = data.Data('uniform', N, 2, 0.0, 2)
    example_ent_unif(dat)

    #exp_err_ent(reps = 100, min_d = 2, max_d = 4, k = 3)

    dat = data.Data('normal', N, 4)
    #test_mutual_information(dat)
    #test_mutual_information_mod(dat, k = 3, show = True)
    exp_err_im(reps = 10, n = 5000, min_d = 2, max_d = 4, k = 3)

if __name__ == "__main__":
    main()
