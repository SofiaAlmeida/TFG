
'''
   Archivo utilizado para realizar pruebas para comparar dos implementaciones de estimadores de la entropía y de la información mutua.
'''

import entropy_estimators as ee
import mutual_info as mi

import random
from numpy import pi
from scipy.special import gamma,psi
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#----------------------------------------------------------------------
## Calcula la entropía de 2 variables aleatorias con ambos estimadores.
# Crea N ejemplos de las variables x e y, estima su entropía con ambos estimadores y la imprime por pantalla.
# @param N número de ejemplos de cada variable a generar.
# @param dx dimensión de la primera variable, x.
# @param dy dimensión de la segunda variable, y.
# @param k número de vecinos a considerar al llamar a los estimadores.
def rd_ent(N, dx, dy, k):
    np.random.seed(0)
    x = np.random.randn(N, dx)
    y = np.random.randn(N, dy)

    print("Entropía de X:")
    print("Estimador 1: ", mi.entropy(x, k = k))
    print("Estimador 2: ", ee.entropy(x))

    print("Entropía de Y:")
    print("Estimador 1: ", mi.entropy(y, k = k))
    print("Estimador 2: ", ee.entropy(y))


#----------------------------------------------------------------------
## Basada en mutual_info.entropy, añade los detalles que diferían de la teoría.
# @param X vector de datos cuya entropía vamos a calcular.
# @param k número de vecinos más cercanos a considerar.
# @return entropía de X.
def entropy_mod(X, k=1):
    # Distance a k-ésimo vecino
    r = 2 * mi.nearest_distances(X, k) 
    n, d = X.shape
    volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1) / 2**d
    
    return (d*np.mean(np.log(2*(r + np.finfo(X.dtype).eps)))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


#---------------------------------------------------------------------
## Adaptación de mutual_info.test_entropy para que se compare también entropy_estimators.entropy con una normal.
# @param n número de ejemplos que crearemos de la variable X.
# @param d dimensión de la variable X.
# @param k número de vecinos a utilizar en los estimadores
def test_entropy(n = 50000, d = 2, k = 3):
    rng = np.random.RandomState(0)
    
    # Creamos la matriz de covarianzas
    P = np.random.randn(d, d)
    C = np.dot(P, P.T)

    # Creamos la variable X
    Y = rng.randn(d, n)
    X = np.dot(P, Y)
    H_th = mi.entropy_gaussian(C) / np.log(2)
    H_est = mi.entropy(X.T, k) 
    H_est2 = ee.entropy(X.T, k)

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
# @param n número de ejemplos a crear de la variable X.
# @param d dimensión de la variable a crear.
# @param k número de vecinos más cercanos a considerar para realizar las estimacioines.
# @param show si vale True imprimirá por pantalla los resultados, si vale False, no.
# @return Diferencia entre la estimación teórica y la estimación realizada por el primer estimador, diferencia entre la estimación teórica y la estimación realizada por el segundo estimador.
def err_entropy(n = 50000, d = 2, k = 3, show = False):
    rng = np.random.RandomState(0)
    # Creamos la matriz de covarianzas
    P = np.random.randn(d, d)
    C = np.dot(P, P.T)
    
    # Creamos la variable X
    Y = rng.randn(d, n)
    X = np.dot(P, Y)
    
    # Calculamos las entropías
    H_th = mi.entropy_gaussian(C) / np.log(2)
    H_est = mi.entropy(X.T, k)
    H_est2 = ee.entropy(X.T, k)

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
## Modificación de pruebas.err_entropy para añadir un tercer estimador.
# Compara tres estimadores de la entropía con la entropía teórica de una variable aleatoria con distribución normal.
# @param n número de muestras a crear de la variable X.
# @param d dimensión de la variable X.
# @param k número de vecinos más cercanos considerados para realizar las estimacioines.
# @return [dif1, dif2, dif3] la diferencia entre la estimación teórica y la calculada por cada uno de los estimadores.
def err_entropy_mod(n = 50000, d = 2, k = 3):
    rng = np.random.RandomState(0)

    # Creamos la matriz de covarianzas
    P = np.random.randn(d, d)
    C = np.dot(P, P.T)
    
    # Creamos la variable aleatoria
    Y = rng.randn(d, n)
    X = np.dot(P, Y)

    # Calculamos la entropía
    H_th = mi.entropy_gaussian(C) / np.log(2)
    H_est = mi.entropy(X.T, k) 
    H_est2 = ee.entropy(X.T, k)
    H_est3 = entropy_mod(X.T, k) / np.log(2)
    
    print("Entropía gaussiana:")
    print("Teórica: ", H_th)
    print("Estimador 1: ", H_est)
    print("Estimador 2: ", H_est2)
    print("Estimador 3: ", H_est3)

    # Calculamos las diferencias
    dif1 = abs(H_th - H_est)
    dif2 = abs(H_th - H_est2)
    dif3 = abs(H_th - H_est3)
    
    return dif1, dif2, dif3

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
            d1, d2 = err_entropy(n, d, k)
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
## Imprime el cálculo de la entropía de una variable aleatoria con distribución uniforme en el intervalor [0,a], teórica y estimada.
# @param a extremo superior del intervalo.
# @param n número de ejemplos a utiliar.
# @param k número de vecinos más cercanos con los que realizar las estimaciones.
def example_ent_unif(a = 2, n = 10000, k = 3):
    x = np.random.uniform(0.0, a, (n, 1))
    H_est = mi.entropy(x, k) 
    H_est2 = ee.entropy(x, k)
    print("Entropía x:")
    print("Teórica: ", np.log(a) / np.log(2))
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
# @param n número de ejemplos a utilizar de cada variable.
# @param k número de vecinos más cercanos a utilizar para realizar la estimación.
def test_mutual_information(n = 50000, k = 3):
    rng = np.random.RandomState(0)
    P = np.random.randn(2, 2)
    C = np.dot(P, P.T)
    U = rng.randn(2, n)
    Z = np.dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    
    MI_est = mi.mutual_information((X, Y), k = k)
    MI_est2 = ee.mi(X, Y, k = k)
    MI_th = (mi.entropy_gaussian(C[0, 0]) / np.log(2)
             + mi.entropy_gaussian(C[1, 1]) / np.log(2)
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
# @param n número de muestras a tomar de las variables.
# @param dimensión de las variables.
# @param k número de vecinos más cercanos a utilizar en los estimadores.
# @param show si True, se imprimen resultados por pantall, si False, no.
# @return Diferencia entre la información mutua teórica y la estimada por la primera implementación, diferencia entre la información mutua teórica y la estimada por la segunda implementación.
def test_mutual_information_mod(n = 50000, d = 2, k = 3, show = False):
    # Creamos las variables
    rng = np.random.RandomState(0)
    P = np.random.randn(d, d)
    C = np.dot(P, P.T)
    U = rng.randn(d, n)
    Z = np.dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    # Estimamos
    MI_est = mi.mutual_information((X, Y), k)
    MI_est2 = ee.mi(X, Y)

    # Información mutua teórica
    MI_th = 0
    for i in range(0, d):
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
            d1, d2 = test_mutual_information_mod(n, d, k)
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
    rd_ent(1000, 20, 3, 3)
    test_entropy(50000, 5, 3)
    err_entropy(50000, 2, 3, True)
    err_entropy_mod()
    
    example_ent()
    example_ent_unif()
    exp_err_ent(reps = 100, min_d = 2, max_d = 4, k = 3)

    test_mutual_information_mod(n = 50000, d = 2, k = 3, show = True)
    exp_err_im(reps = 25, n = 50000, min_d = 2, max_d = 4, k = 3)
#rd_ent_mi()
#exp_err_ent(reps = 20, k = 3, max_d = 15)
# exp_err_im(reps = 25, k = 3, max_d = 20)


if __name__ == "__main__":
    main()
