import sys
import entropy_estimators as ee
sys.path.insert(1, '../../ead9898bd3c973c40429/')
import mutual_info as mi

import random
from numpy import pi
from scipy.special import gamma,psi
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#----------------------------------------------------------------------
# Calcula la entropía de 2 variables aleatorias con ambos estimadores
def rd_ent():
    np.random.seed(0)
    x = np.random.randn(1000, 20)
    y = np.random.randn(1000, 3)

    print("Entropía de X:")
    print("Estimador 1: ", mi.entropy(x, k = 3))
    print("Estimador 2: ", ee.entropy(x))

    print("Entropía de Y:")
    print("Estimador 1: ", mi.entropy(y, k = 3))
    print("Estimador 2: ", ee.entropy(y))


#----------------------------------------------------------------------
# Basada en mutual_info.entropy, añade los detalles que diferían de la teoría
def entropy_mod(X, k=1):
    ''' Returns the entropy of the X.

    Parameters
    ===========

    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed

    k : int, optional
        number of nearest neighbors for density estimation

    Notes
    ======

    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''

    # Distance to kth nearest neighbor
    r = 2 * mi.nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1) / 2**d
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.

    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return (d*np.mean(np.log(2*(r + np.finfo(X.dtype).eps)))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


#---------------------------------------------------------------------
# Adaptación de mutual_info.test_entropy para que se compare también entropy_estimators.entropy con una normal 
def test_entropy(d, k = 3):
    # Testing against correlated Gaussian variables
    # (analytical results are known)
    # Entropy of a d-dimensional gaussian variable
    rng = np.random.RandomState(0)
    n = 50000
    P = np.random.randn(d, d)    C = np.dot(P, P.T)
    Y = rng.randn(d, n)
    X = np.dot(P, Y)
    H_th = mi.entropy_gaussian(C) / np.log(2)
    H_est = mi.entropy(X.T, k) / np.log(2)
    H_est2 = ee.entropy(X.T, k)
    print("Entropía gaussiana:")
    print("Teórica: ", H_th)
    print("Estimador 1: ", H_est)
    print("Estimador 2: ", H_est2)
    # Our estimated entropy should always be less that the actual one
    # (entropy estimation undershoots) but not too much
    np.testing.assert_array_less(H_est, H_th)
    np.testing.assert_array_less(H_est2, H_th)
    np.testing.assert_array_less(.9*H_th, H_est)
    np.testing.assert_array_less(.9*H_th, H_est2)
    

#---------------------------------------------------------------------
# Modificación de test_entropy, devuelve la diferencia entre la entropía teórica y las estimadas
def err_entropy(d, k, show = False):
    # Testing against correlated Gaussian variables
    # (analytical results are known)
    # Entropy of a 3-dimensional gaussian variable
    rng = np.random.RandomState(0)
    n = 50000
    P = np.random.randn(d, d)
    #P = np.array([[1, 0, 0], [0, 1, .5], [0, 0, 1]])
    C = np.dot(P, P.T)
    Y = rng.randn(d, n)
    X = np.dot(P, Y)
    H_th = mi.entropy_gaussian(C) / np.log(2)
    H_est = mi.entropy(X.T, k)
    H_est2 = ee.entropy(X.T, k)

    if(show):
        print("Entropía gaussiana:")
        print("Teórica: ", H_th)
        print("Estimador 1: ", H_est)
        print("Estimador 2: ", H_est2)
    
    dif1 = abs(H_th - H_est)
    dif2 = abs(H_th - H_est2)
    return dif1, dif2

#---------------------------------------------------------------------
# Modificación de err_entropy para añadir un tercer estimador
def err_entropy_mod(d, k = 3):
    # Testing against correlated Gaussian variables
    # (analytical results are known)
    # Entropy of a 3-dimensional gaussian variable
    rng = np.random.RandomState(0)
    n = 50000
    P = np.random.randn(d, d)
    #P = np.array([[1, 0, 0], [0, 1, .5], [0, 0, 1]])
    C = np.dot(P, P.T)
    Y = rng.randn(d, n)
    X = np.dot(P, Y)
    H_th = mi.entropy_gaussian(C) / np.log(2)
    H_est = mi.entropy(X.T, k) / np.log(2)
    H_est2 = ee.entropy(X.T, k)
    H_est3 = entropy_mod(X.T, k) / np.log(2)
    
    print("Entropía gaussiana:")
    print("Teórica: ", H_th)
    print("Estimador 1: ", H_est)
    print("Estimador 2: ", H_est2)
    print("Estimador 3: ", H_est3)
    
    dif1 = abs(H_th - H_est)
    dif2 = abs(H_th - H_est2)
    dif3 = abs(H_th - H_est3)
    
    return dif1, dif2, dif3

#---------------------------------------------------------------------
# Calculo de entropía para un caso concreto
# No va a dar resultados correctos porque es una variable discreta y los estimadores son para variables continuas
def example_ent():
    x = np.random.choice([1.0,2.0,3.0,4.0], (1000, 1) , p=[0.5, 0.25, 0.125, 0.125])
    y = np.random.choice([1.0,2.0,3.0,4.0], (1000, 1) , p=[0.25, 0.25, 0.25, 0.25])
    H_est = mi.entropy(x, k=5) / np.log(2)
    H_est2 = ee.entropy(x, k=5)
    print("Entropía x:")
    print("Teórica: ", 7.0/4.0)
    print("Estimador 1: ", H_est)
    print("Estimador 2: ", H_est2)
    H_est = mi.entropy(y, k=5) / np.log(2)
    H_est2 = ee.entropy(y, k=5)
    print("Entropía y:")
    print("Teórica: ", 2.0)
    print("Estimador 1: ", H_est)
    print("Estimador 2: ", H_est2)

#---------------------------------------------------------------------
# Entropía de una distribución uniforme, teórica y estimada
def example_entu():
    a = 2
    x = np.random.uniform(0.0, a, (1000, 1))
    H_est = mi.entropy(x, k=5) / np.log(2)
    H_est2 = ee.entropy(x, k=5)
    print("Entropía x:")
    print("Teórica: ", np.log(a) / np.log(2))
    print("Estimador 1: ", H_est)
    print("Estimador 2: ", H_est2)

#---------------------------------------------------------------------
# Plot of two arrays
def plot_err(x, y, max_d):
    plt.plot(range(2, max_d + 1), x, label = "Error estimador 1")
    plt.plot(range(2, max_d + 1), y, label = "Error estimador 2")
    plt.legend(loc="upper left")
    plt.xlabel("d")
    plt.show()
    
    
#---------------------------------------------------------------------
# Calcula la diferencia entre la entropía teórica y las estimadas
# para dimensiones desde 2 hasta max_d, el error se calcula como la media en las reps repeticiones de la ejecución para cada dimensión
def exp_err_ent(reps = 100, max_d = 9, k = 3):    
    err1 = np.array([])
    err2 = np.array([])
    
    for d in range(2, max_d + 1):
        dif1 = np.array([])
        dif2 = np.array([])
        for i in range(0, reps):
            d1, d2 = err_entropy(d, k)
            dif1 = np.append(dif1, d1)
            dif2 = np.append(dif2, d2)

        e1 = np.mean(dif1)
        e2 = np.mean(dif2)
        err1 = np.append(err1, e1)
        err2 = np.append(err2, e2)

        print("d = ", d)
        print("Error estimador 1: ", e1)
        print("Error estimador 2: ", e2)

    plot_err(err1, err2, max_d)

#-----------------------------------------------------------------------
# Adaptación de mutual_information.test_mutual_information para incluir a entropy_estimators.mi
def test_mutual_information():
    # Mutual information between two correlated gaussian variables
    # Entropy of a 2-dimensional gaussian variable
    n = 50000
    rng = np.random.RandomState(0)
    P = np.random.randn(2, 2)
    #P = np.array([[1, 0], [0.5, 1]])
    C = np.dot(P, P.T)
    U = rng.randn(2, n)
    Z = np.dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    # in bits
    MI_est = mi.mutual_information((X, Y), k=3)
    MI_est2 = ee.mi(X, Y)
    MI_th = (mi.entropy_gaussian(C[0, 0]) / np.log(2)
             + mi.entropy_gaussian(C[1, 1]) / np.log(2)
             - mi.entropy_gaussian(C) / np.log(2)
            )
    # Our estimator should undershoot once again: it will undershoot more
    # for the 2D estimation that for the 1D estimation

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
# Modificación de test_mutual_information para permitir que las variables tengan la dimensión d indicada
def test_mutual_information_mod(d, k, show = False):
    # Mutual information between two correlated gaussian variables
    # Entropy of a d-dimensional gaussian variable
    n = 50000
    rng = np.random.RandomState(0)
    P = np.random.randn(d, d)
    C = np.dot(P, P.T)
    U = rng.randn(d, n)
    Z = np.dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    # in bits
    MI_est = mi.mutual_information((X, Y), k)
    MI_est2 = ee.mi(X, Y)

    # Theoretical MI
    MI_th = 0
    for i in range(0, d):
        MI_th += mi.entropy_gaussian(C[i, i]) / np.log(2)
  
    MI_th -=  mi.entropy_gaussian(C) / np.log(2)

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
# Calcularemos los errores obtenidos con ambos estimadores
# Para dimensiones de 2 hasta max_d, repitiendo reps veces el cálculo por dimensión y devolviendo la media de las diferencias con la entropía teórica
def exp_err_im(reps = 100, max_d = 9, k = 3):    
    err1 = np.array([])
    err2 = np.array([])
    
    for d in range(2, max_d + 1):
        dif1 = np.array([])
        dif2 = np.array([])
        for i in range(0, reps):
            d1, d2 = test_mutual_information_mod(d, k)
            dif1 = np.append(dif1, d1)
            dif2 = np.append(dif2, d2)

        e1 = np.mean(dif1)
        e2 = np.mean(dif2)
        err1 = np.append(err1, e1)
        err2 = np.append(err2, e2)

        print("d = ", d)
        print("Error estimador 1: ", e1)
        print("Error estimador 2: ", e2)

    plot_err(err1, err2, max_d)

    
#-----------------------------------------------------------------------
#rd_ent_mi()
#exp_err_ent(reps = 20, k = 3, max_d = 15)
exp_err_im(reps = 25, k = 3, max_d = 20)
input("ENTER TO CONTINUE")


#example_entc()

input("ENTER TO CONTINUE")

