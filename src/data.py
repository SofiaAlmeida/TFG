import numpy as np
import random


## Clase para generar los datos.
#
# Permite generar datos según una distribución normal, uniforme o de forma aleatoria.
class Data:
    ## Constructor
    # @param dist indica la distribución a utilizar, los posibles valores son 'random','normal' y 'uniform'.
    # @param N número de muestras a generar.
    # @param d dimensión de las muestras.
    # @param a cuando dist = 'uniform', este valor indicará el extremo inferior del intervalo.
    # @param b cuando dist = 'uniform', este valor indicará el extremo superior del intervalo.
    def __init__(self, dist, N, d, a = None, b = None):
        if dist ==  'random':
            self.N = N
            self.d = d
            self.X =  np.random.randn(N, d)
            self.a = self.b = None

        elif dist == 'normal':
            self.N = N
            self.d = d
            P = np.random.randn(d, d) 
            self.C = np.dot(P, P.T)
            Y = np.random.randn(d, N)
            self.X = np.dot(P, Y).T
            self.a = self.b = None

        elif dist == 'uniform':
            self.N = N
            self.d = d
            self.a = a
            self.b = b
            self.X = np.random.uniform(a, b, (N, d))
            self.C = None
