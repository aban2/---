from search_policy import bisect, golden_section, fibonacci, dichotomous


def forward(x):
    return f[2]+f[1]*x+f[0]*x*x;


def get_gradient(x):
    return 2*f[0]*x+f[1];


if __name__ == '__main__':
    # init data
    f = [3, -21.6, -1] # 分别代表二次项，一次项，常数项
    a0, b0 = 0, 25
    acc = 0.08
    mid, num_iters = fibonacci(a0, b0, acc, forward, get_gradient)
    print('Finished after', num_iters, 'iterations')
    print('x:', mid, 'f(x):', forward(mid), 'grad:', get_gradient(mid))
