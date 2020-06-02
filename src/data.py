class Data:
    def __init__(self, dist, N, d, a, b):
        if dist ==  'random':
            self.N = N
            self.d = d
            self.X =  np.random.randn(N, dx)
            self.C = self.a = self.b = NULL

        elif dist == 'normal':
            self.N = N
            self.d = d
            P = np.random.randn(d, d) 
            self.C = np.dot(P, P.T)
            Y = rng.randn(d, n) 
            self.X = np.dot(P, Y)
            self.a = self.b = NULL

        elif dist == 'uniform':
            self.N = N
            self.d = d
            self.a = a
            self.b = b
            self.X = np.random.uniform(a, b, (n, d))
            self.C = NULL
