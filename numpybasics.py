import numpy as np
arr = np.arange(12)
reshaped = np.reshape(arr,[2,3,2])

flattened = reshaped.flatten()
transposed = np.transpose(reshaped,axes=(1,2,0))

zeros_arr = np.zeros(5)
ones_arr = np.ones_like(transposed)

points = np.linspace(-3.5, 1.5, num=101)