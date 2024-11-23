import scipy.optimize
def codigo(points, x0 = [1,1]):
    def funcao(parametros, points):
        sum = 0
        a, b = parametros[0], parametros[1]
        for i in range (len(points)):
            sum+= abs(a*(points[i][0]) + b - points[i][1])
        return sum

    
    
    result = scipy.optimize.minimize(funcao, x0, args = (points))
    print(f'a = {result.x[0]} and b = {result.x[1]}')
pontos = [(1, 2), (2, 4), (3, 6), (4, 8)]

x0 = [1,1]

codigo(pontos)

import numpy as np

def generate_points(m):
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
def rodando(lista):
    for i in lista:
        c = generate_points(i)
        c1 = []
        for j in range(i):
            c2 = []
            c2.append(c[0][j])
            c2.append(c[1][j])
            c1.append(c2)
        codigo(c1)

rodando(listinha)
print("\n\n\n\n")

def outro_codigo(points, x0 = [1,1]):
    def outra_funcao(parametros, points):
        sum = 0
        a, b = parametros[0], parametros[1]
        for i in range (len(points)):
            sum+= (a*(points[i][0]) + b - points[i][1])**2
        return sum

    
    
    result = scipy.optimize.minimize(outra_funcao, x0, args = (points))
    print(f'a = {result.x[0]} and b = {result.x[1]}')

def outro_rodando(lista):
    for i in lista:
        c = generate_points(i)
        c1 = []
        for j in range(i):
            c2 = []
            c2.append(c[0][j])
            c2.append(c[1][j])
            c1.append(c2)
        outro_codigo(c1)
outro_rodando(listinha)
