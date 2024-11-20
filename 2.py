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
        
    
    def predict(self, coefficients, X):
        return np.dot(X, coefficients)

pontos = [(1, 1), (2, 4), (3, 9), (4, 16)]
model = regression_poly(pontos)
coefs = model.aprox(2)
print(coefs)      

def predict(max_degree):
    erros = []
    for i in range(max_degree+1):
          coefici, X, Y = model.aprox(i)
          predictions = model.predict(coefici, X)
          mse = np.mean((Y - predictions)**2)
          erros.append(mse)
    return erros
error = predict(10)

melhor_grau = np.argmin(error) 
print(f'Melhor grau ta sendo: {melhor_grau}')
