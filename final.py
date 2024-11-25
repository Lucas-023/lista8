#Questão 1
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
if __name__ == '__main__':
         
    pontos = [(1, 1), (2, 4), (3, 9), (4, 16)]
    model = regression_poly(pontos)
    coefs = model.aprox(2)
    print(coefs)        

#Questão 2
'''A estrategia aqui implementada foi de definir um grau máximo para o polinomio e testar todos os graus menores ou iguais
 a esse de inteiros nao negativos e então analisar o que tivesse menor erro quadrático'''
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
        return coeficientes, X, Y
        
    
    def predict(self, coefficients, X):
        return np.dot(X, coefficients)
   

def prever(max_degree,z:regression_poly):
    erros = []
    for i in range(max_degree+1):
          coefici, X, Y = z.aprox(i)
          predictions = z.predict(coefici, X)
          mse = np.mean((Y - predictions)**2)
          erros.append(mse)
    return erros

if __name__ == '__main__':#O metodo se mostra preciso em todos os testes retornando o grau de maior precisão, é possível observar que foram testadas funções exatas para que tivessemos certeza de qual seria a resposta certa sendo essas as funcções y=x, y=x², y=x³ e y=x³ +3
    pontos1 = [(1, 1), (2, 2), (3, 3), (4, 4)]
    pontos2 = [(1, 1), (2, 4), (3, 9), (4, 16)]
    pontos3 = [(1, 1), (2, 8), (3, 27), (4, 64)]
    pontos4 = [(1, 4), (2, 11), (3, 30), (4, 67)]




    modelo1 = regression_poly(pontos1)
    modelo2 = regression_poly(pontos2)
    modelo3 = regression_poly(pontos3)
    modelo4 = regression_poly(pontos4)

    erro1 = prever(10, modelo1)
    erro2 = prever(10, modelo2)
    erro3 = prever(10, modelo3)
    erro4 = prever(10, modelo4)


    print(np.argmin(erro1))
    print(np.argmin(erro2))
    print(np.argmin(erro3))
    print(np.argmin(erro4))

#Questão 3
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np


def absolute(points, x0 = [1,1]):#função para o item a
    def func_abs(parametros, points):#funcao para ser minimizada
        sum = 0
        a, b = parametros[0], parametros[1]
        for i in range (len(points)):
            sum+= abs(a*(points[i][0]) + b - points[i][1])
        return sum

    
    
    result = scipy.optimize.minimize(func_abs, x0, args = (points))
    a,b = result.x
    x = []
    y = []
    for i in points:
        x.append(i[0])
        y.append(i[1])
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', markeredgewidth=2)

    data_x = np.linspace(min(x), max(x), 100)
    ax.plot(data_x, data_x*a + b)

    plt.show()#plot necessário para o item d

    return (f'a = {result.x[0]} and b = {result.x[1]}')

def quad(points, x0 = [1,1]):#função item, c
    def func_quad(parametros, points):#função a ser minimizada
        sum = 0
        a, b = parametros[0], parametros[1]
        for i in range (len(points)):
            sum+= (a*(points[i][0]) + b - points[i][1])**2
        return sum


    result = scipy.optimize.minimize(func_quad, x0, args = (points))

    a,b = result.x
    x = []
    y = []
    for i in points:
        x.append(i[0])
        y.append(i[1])
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', markeredgewidth=2)

    data_x = np.linspace(min(x), max(x), 100)   
    ax.plot(data_x, data_x*a + b)

    plt.show()#plot necessário para o item d

    return(f'a = {result.x[0]} and b = {result.x[1]}')


if __name__ == '__main__':
    pontos = [(1, 2), (2, 4), (3, 6), (4, 8)]

    x0 = [1,1]
    print(absolute(pontos))
    print(quad(pontos))

    def generate_points(m):#gerando pontos para o item b
        np.random.seed(1)
        a = 6
        b = -3
        x = np.linspace(0, 10, m)
        y = a*x + b + np.random.standard_cauchy(size=m)

        return (x,y)

    def save_points(points, path = 'test_points.txt' ):
        with open(path, 'wt') as f:
            for x, y in zip(points[0], points[1]):
                f.write(f'{x} {y}\n')

    listinha = [64, 128, 256, 512, 1024]
    def rodando(lista):#testando e utilizando os pontos do item b na funcao do item a
        for i in lista:
            c = generate_points(i)
            c1 = []
            for j in range(i):
                c2 = []
                c2.append(c[0][j])
                c2.append(c[1][j])
                c1.append(c2)
            print(absolute(c1))

    rodando(listinha)
    print("\n\n")



    def outro_rodando(listinha):#testando e utilizando os pontos do item b na funcao do item c
        for i in listinha:
            c = generate_points(i)
            c1 = []
            for j in range(i):
                c2 = []
                c2.append(c[0][j])
                c2.append(c[1][j])
                c1.append(c2)
            print(quad(c1))
            
    outro_rodando(listinha)

    """LETRA E
    O erro modular é interessante para 'mover a função', ou seja comparar funcções em diferentes locais do plano,
    enquanto o erro quadrático aumenta de forma quadrática a distância de pontos 
    muito dispersos e quando estão em locais diferentes do plano. Porém, o erro quadrático é diferenciável, e não utiliza de métodos
    numéricos para resolver o mínimo, mas sim de um sistema o que facilita consideravelmente sua implementação.
    """
