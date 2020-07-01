'''
    data.py
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
