import numpy.linalg
import numpy as np
class regression_poly:
    def __init__(self, points):
        self.points = points

    def aprox(self, d):
        X = []
        Y = []
        for i in self.points:
            Y.append([i[1]])

        for j in self.points:
            x = []
            for k in range(d+1):
                    x.append(pow(j[0], k))
            X.append(x)

        X = np.array(X)
        Y = np.array(Y)

        coeficientes = numpy.linalg.lstsq(X, Y, rcond=None)[0]

        coeficientes = np.round(coeficientes, decimals=10)
        return coeficientes
    
pontos = [(1, 1), (2, 4), (3, 9), (4, 16)]
model = regression_poly(pontos)
coefs = model.aprox(1)
print(coefs)        
