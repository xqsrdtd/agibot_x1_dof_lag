import numpy as np
a=np.array([1,2,3])
b=np.array([0,-3,2])
mask = np.abs(a) < np.abs(b)
a[mask] = b[mask]
print(a)
