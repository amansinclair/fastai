import numpy as np
import time

I = 30
J = 40
K = 50

a = np.random.randn(I, J)
b = np.random.randn(J, K)


def how_long(func, *args, n_times=100):
    t = time.time()
    for i in range(n_times):
        func(*args)
    return str(1000 * ((time.time() - t) / n_times)) + " ms"


def matmul(a, b):
    c = np.zeros((a.shape[1], b.shape[1]))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(b.shape[1]):
                c[j, k] += a[i, j] * b[j, k]
    return c


print(np.matmul(a, b).sum())
print(matmul(a, b).sum())

print(how_long(matmul, a, b))
print(how_long(np.matmul, a, b))
