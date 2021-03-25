import numpy as np
from collections import Counter

# 通过矩阵变换形成单位矩阵
def identitify(a, b, c, cb, xb, basis_in):
    # 找到入口基所在行
    zero_row = xb.index(basis_in)
    
    # 剩余行除以零行把对应元素变成0
    for i in range(a.shape[0]):
        if i == zero_row: continue
        mul = a[i][basis_in] / a[zero_row][basis_in]
        a[i] -= a[zero_row]*mul
        b[i] -= b[zero_row]*mul

    b[zero_row] /= a[zero_row][basis_in]
    a[zero_row] /= a[zero_row][basis_in]
    cb[zero_row] = c[basis_in]

def simplex(a, b, c, xb, cb):
    _inf = 1 << 60

    # 循环更新基
    while True:
        print(a)
        # 计算检验数，找到入口基
        checkers = c.T - np.dot(cb.T, a)
        print('checkers', checkers)
        for idx, num in enumerate(checkers[0]):
            if num > 0:
                basis_in = idx
                break
        # basis_in = np.argmax(checkers)
        
        # 计算当前价值
        value = np.dot(cb.T, b)
        print('value', value)
        
        # 若所有检验数<=0，停止算法
        if np.max(checkers) <= 0:
            break

        # 计算theta，找到出口基
        thetas = b / a[:, basis_in].reshape(a.shape[0],1)
        thetas[thetas < 0] = _inf
        idx = np.argmin(thetas)
        basis_out = list(xb)[idx]

        # 更新a中的单位矩阵
        out_idx = xb.index(basis_out)
        xb[out_idx] = basis_in
        identitify(a, b, c, cb, xb, basis_in)

        # 打印单轮信息
        print('---update--')
        print(basis_in+1, 'in and', basis_out+1, 'out')
        print('b:', b)
        print('cb:', cb)
        print('theta:', thetas)
        print()

# 对数据预处理，如果没有初始单位阵，需要用大M法（1）或两阶段法（2）处理
def driver(a, b, c, method=1):
    # 识别单位矩阵，找到初始解
    xb = []
    idxes = set() # 每行都得有1，idxes记录了含有1的行（该列其他元素为0）
    rows, cols = a.shape # 行数
    for i in range(cols):
        col = a[:,i]
        counter = Counter(col)
        if counter[1] == 1 and counter[0] == rows-1:
            pos = list(col).index(1)
            if pos not in idxes:
                idxes.add(pos)
                xb.append(i)
    
    # 检验，若存在单位矩阵，正常操作，若不存在，使用大M法或两阶段法处理
    if len(idxes) < rows:
        num_new = rows - len(idxes)
        # 加变量
        for i in range(rows):
            if i not in idxes:
                xb.append(c.shape[0])
                col = np.zeros((a.shape[0], 1))
                col[i] = 1
                a = np.hstack((a, col))
                c = np.vstack((c, np.zeros((1, 1))))

        # 使用大M法
        if method == 1:
            M = 50
            for i in range(len(c)-1, len(c)-rows-1, -1):
                c[i] = -M
            simplex(a, b, c, xb, c[xb])
        
        # 使用两阶段法
        elif method == 2:
            print('--- first phase ----')
            p1_c = np.zeros(c.shape)
            for i in range(len(c)-1, len(c)-rows-1, -1):
                p1_c[i] = -1
            simplex(a, b, p1_c, xb, p1_c[xb])
            print()

            print('----second phase ----')
            simplex(a[:,:-num_new], b, c[:-num_new], xb, c[xb])

    else:
        simplex(a, b, c, xb, c[xb])

a = np.array([[2, -1, 3, 1], [1, 2, 4, 0]], dtype=np.float64)
b = np.array([[30, 40]], dtype=np.float64).T
c = np.array([[4,  2, 8, 0]]).T

driver(a, b, c, 1)