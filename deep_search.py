import numpy as np
from search_policy import momentum, RMSprop, Adadelta


def f(x):
    return x[0]**2+x[1]**2-x[0]*x[1]-10*x[0]-4*x[1]+60


def grad(x):
    return np.array([[2*x[0]-x[1]-10, 2*x[1]-x[0]-4]]).T


if __name__ == '__main__':
    x0 = [0, 0]
    eps = 0.001
    x, num_iters = momentum(x0, eps, f, grad)
    print('Finished after', num_iters, 'iterations')
    print('f(x):%.6f' % (f(x)))
    print(x)
