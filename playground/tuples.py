import numpy as np

a = np.array([[12, 1000], [23, 400], [45, 600]])

print(a[a[:, 1] > 700][:, 0])
