from math import sqrt


def forward(f, x):
    return f[2]+f[1]*x+f[0]*x*x;


def get_gradient(f, x):
    return 2*f[0]*x+f[1];


# 二分法
def bisect(f, a0, b0, acc):
    l, r, num_iters = a0, b0, 0
    while r-l >= acc:
        num_iters += 1
        mid = (l+r) / 2
        gradient = get_gradient(f, mid)
        if gradient == 0: return mid, num_iters
        elif gradient < 0: l = mid
        else: r = mid
    return (l+r)/2, num_iters


# 黄金分割法
def gold_split(f, a0, b0, acc):
    # 选取初始点
    l, r, t = a0, b0, (sqrt(5)-1)/2
    m, n = l+(1-t)*(r-l), l+t*(r-l)
    num_iters = 0

    while r-l >= acc:
        num_iters += 1
        if forward(f, m) > forward(f, n):
            l = m
            m = n
            n = r - (m-l)
        else:
            r = n
            n = m
            m = l + (r-n)
    return (l+r)/2, num_iters


# 斐波那契搜索
def fibonacci(f, a0, b0, acc):
    fibs = [0, 1, 1, 2]
    l, r, num_iters = a0, b0, 0

    while r-l >= acc:
        num_iters += 1
        fibs.append(fibs[-1]+fibs[-2])
        m, n = l + (r - l) * (fibs[-3] / fibs[-1]), l + (r - l) * (fibs[-2] / fibs[-1])
        if forward(f, m) > forward(f, n): l = m
        else: r = n

    return (l+r)/2, num_iters


def dichotomous(f, a0, b0, acc):
    l, r, num_iters = a0, b0, 0
    float_judge = 1e-5

    while r-l >= acc:
        num_iters += 1
        mid = (l+r) / 2
        m, n = mid-acc, mid+acc
        if abs(m-l) < float_judge and abs(n-r) < float_judge: break
        if(forward(f, m) > forward(f, n)): l = m
        else: r = n

    return (l+r)/2, num_iters

if __name__ == '__main__':
    # init data
    f = [3, -21.6, -1] # 分别代表二次项，一次项，常数项
    a0, b0 = 0, 25
    acc = 0.08
    mid, num_iters = fibonacci(f, a0, b0, acc)
    print('Finished after', num_iters, 'iterations')
    print('x:', mid, 'f(x):', forward(f, mid), 'grad:', get_gradient(f, mid))

