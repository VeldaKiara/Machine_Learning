import numpy as np 
arr = np.array([[-0.5,0.8,-0.1], [0.0,-1.2,1.3]])
arr2 = np.array([[1.2,3.1],[1.2,0.3],[1.5,2.2]])

multiplied = arr * np.pi
added = arr + multiplied
squared = added ** 2

exponential = np.exp(squared)
logged = np.log(arr2)

matmul1 = np.matmul(logged,exponential)
matmul2 = np.matmul(exponential, logged)

