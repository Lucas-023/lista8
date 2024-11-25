import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np


def absolute(points, x0 = [1,1]):
    def func_abs(parametros, points):
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

    plt.show()

    return (f'a = {result.x[0]} and b = {result.x[1]}')

def quad(points, x0 = [1,1]):
    def func_quad(parametros, points):
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

    plt.show()

    return(f'a = {result.x[0]} and b = {result.x[1]}')


if __name__ == 'main':
    pontos = [(1, 2), (2, 4), (3, 6), (4, 8)]

    x0 = [1,1]
    print(absolute(pontos))
    print(quad(pontos))

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
            print(absolute(c1))

    rodando(listinha)
    print("\n\n\n\n")



    def outro_rodando(listinha):
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

