from search_policy import gold_stein, wolfe_powell, bisect, golden_section


def forward(step_size):
    inputs = [x[i]+step_size*d[i] for i in range(len(x))]
    return 100*(inputs[1]-inputs[0]**2)**2 + (1-inputs[0])**2


def get_gradient(step_size):
    inputs = [x[i]+step_size*d[i] for i in range(len(x))]
    grad_fn = [400*inputs[0]**3 - 400*inputs[0]*inputs[1] + 2*inputs[0] - 2,
               200*inputs[1] - 200*inputs[0]*inputs[0]]
    return grad_fn[0]*d[0] + grad_fn[1]*d[1]


if __name__ == '__main__':
    d = [1, 1]
    x = [-1, 1]
    step_size, num_iters = gold_stein(1, forward, get_gradient)
    print('Approximate Search Finished after', num_iters, 'iterations')
    print('step_size:%.5f f(x):%.2f grad:%.2f' % (step_size, forward(step_size), get_gradient(step_size)))
    print()

    step_size, num_iters = bisect(0, 1, 0.0005, forward, get_gradient)
    new_x = [x[i]+step_size*d[i] for i in range(len(x))]
    print('Precise Search Finished after', num_iters, 'iterations')
    print('step_size:%.5f f(x):%.2f grad:%.2f' % (step_size, forward(step_size), get_gradient(step_size)))
