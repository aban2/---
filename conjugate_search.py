import numpy as np
from search_policy import flecher_reeves


def f3(x):
    return x[0]**2+x[1]**2-x[0]*x[1]-10*x[0]-4*x[1]+60


def grad3(x):
    return np.array([[2*x[0]-x[1]-10, 2*x[1]-x[0]-4]]).T


def f1(x):
    return x[0]-x[1]+2*x[0]**2+2*x[0]*x[1]+x[1]**2


def grad1(x):
    return np.array([[1+4*x[0]+2*x[1], -1+2*x[0]+2*x[1]]]).T


if __name__ == '__main__':
    x0 = [0, 0]
    eps = 1e-6
    fn, grad_fn = f1, grad1
    x, num_iters = flecher_reeves(x0, eps, fn, grad_fn)
    print('Finished after', num_iters, 'iterations')
    print('f(x):%.6f' % (fn(x)))
    print(x)

