from math import sqrt


def dot(x, y):
    sum0 = 0
    for i in range(len(x)): sum0 += x[i]*y[i]
    return sum0


# 二分法
def bisect(a0, b0, acc, fn, grad_fn):
    l, r, num_iters = a0, b0, 0
    while r-l >= acc:
        num_iters += 1
        mid = (l+r) / 2
        gradient = grad_fn(mid)
        if gradient == 0: return mid, num_iters
        elif gradient < 0: l = mid
        else: r = mid
    return (l+r)/2, num_iters


# 黄金分割法
def golden_section(a0, b0, acc, fn, grad_fn):
    # 选取初始点
    l, r, t = a0, b0, (sqrt(5)-1)/2
    m, n = l+(1-t)*(r-l), l+t*(r-l)
    num_iters = 0

    while r-l >= acc:
        num_iters += 1
        if fn(m) > fn(n):
            l = m
            m = n
            n = r - (m-l)
        else:
            r = n
            n = m
            m = l + (r-n)
    return (l+r)/2, num_iters


# 斐波那契搜索
def fibonacci(a0, b0, acc, fn, grad_fn):
    fibs = [0, 1, 1, 2]
    l, r, num_iters = a0, b0, 0

    while r-l >= acc:
        num_iters += 1
        fibs.append(fibs[-1]+fibs[-2])
        m, n = l + (r - l) * (fibs[-3] / fibs[-1]), l + (r - l) * (fibs[-2] / fibs[-1])
        if fn(m) > fn(n): l = m
        else: r = n

    return (l+r)/2, num_iters


def dichotomous(a0, b0, acc, fn, grad_fn):
    l, r, num_iters = a0, b0, 0
    float_judge = 1e-5

    while r-l >= acc:
        num_iters += 1
        mid = (l+r) / 2
        m, n = mid-acc, mid+acc
        if abs(m-l) < float_judge and abs(n-r) < float_judge: break
        if(fn(m) > fn(n)): l = m
        else: r = n

    return (l+r)/2, num_iters


def gold_stein(step_size, fn, grad_fn):
    rou, alpha, beta, num_iters = 0.08, 1.5, 0.5, 0

    grad0, f0 = grad_fn(0), fn(0)
    while True:
        num_iters += 1
        # 选取的d的方向和梯度方向相反，所以此处用梯度上升
        upper_bound = rou * grad0 * step_size
        lower_bound = (1 - rou) * grad0 * step_size
        diff = fn(step_size) - f0

        if diff < lower_bound: step_size *= alpha
        elif diff > upper_bound: step_size *= beta
        else: return step_size, num_iters


def wolfe_powell(step_size, fn, grad_fn):
    rou, sigma, alpha, beta, num_iters = 0.08, 1.25, 1.5, 0.5, 0

    grad0, f0 = grad_fn(0), fn(0)
    while True:
        num_iters += 1
        # 选取的d的方向和梯度方向相反，所以此处用梯度上升
        upper_bound = rou * grad0 * step_size
        diff = fn(step_size) - f0
        new_grad = grad_fn(step_size)

        if diff > upper_bound: step_size *= beta
        elif new_grad < sigma*grad0: step_size *= alpha
        else: return step_size, num_iters
