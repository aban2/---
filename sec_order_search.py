import numpy as np
from search_policy import DFP, BFGS


def f0(x):
    return (x[0]-2)**4+(x[0]-2*x[1])**2


def grad0(x):
    return np.array([[4*(x[0]-2)**3+2*(x[0]-2*x[1]), 2*(x[0]-2*x[1])*(-2)]]).T


def f1(x):
    return 10*x[0]**2+x[1]**2


def grad1(x):
    return np.array([[20*x[0], 2*x[1]]]).T


def f2(x):
    return x[0]**2+4*x[1]**2-4*x[0]-8*x[1]


def grad2(x):
    return np.array([[2*x[0]-4, 8*x[1]-8]]).T


def f3(x):
    return x[0]**2+x[1]**2-x[0]*x[1]-10*x[0]-4*x[1]+60


def grad3(x):
    return np.array([[2*x[0]-x[1]-10, 2*x[1]-x[0]-4]]).T


if __name__ == '__main__':
    h0 = np.array([[1, 0], [0, 1]], dtype=np.float64)
    x0 = [0, 0]
    eps = 0.001
    fn, grad_fn = f3, grad3
    x, num_iters = DFP(x0, h0, eps, fn, grad_fn)
    print('Finished after', num_iters, 'iterations')
    print('f(x):%.6f' % (fn(x)))
    print(x)
