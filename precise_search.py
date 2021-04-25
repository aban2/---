from search_policy import bisect, golden_section, fibonacci, dichotomous


def forward(inputs):
    return f[2]+f[1]*inputs+f[0]*inputs**2


def get_gradient(inputs):
    return 2*f[0]*inputs+f[1]


if __name__ == '__main__':
    # init data
    f = [3, -21.6, -1] # 分别代表二次项，一次项，常数项
    a0, b0 = 0, 25
    acc = 0.08
    x, num_iters = dichotomous(a0, b0, acc, forward, get_gradient)
    print('Finished after', num_iters, 'iterations')
    print('step_size:%.5f f(x):%.3f grad:%.2f' % (x, forward(x), get_gradient(x)))
