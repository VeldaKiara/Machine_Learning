import numpy as np
#using nan
arr = np.array([np.nan,2,3,4,5])

#using copy
arr2=arr.copy()
arr2[0]=10

#Using astype
float_arr = np.array([1,5.4,3])
float_arr2 = arr2.astype(np.float32)

#using dtype
matrix = np.array([[1,2,3],[4,5,6]],  dtype=np.float32)