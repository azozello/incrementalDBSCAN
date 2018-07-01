import numpy as np

a = np.array([[2.22, 14.88], [3.22, 6.66]])
b = np.array([[3.14, 1.61]])

print(a)
print(b)

c = np.concatenate((a, b))

print()

print(c)
