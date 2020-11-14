'''
Set elem equal to the third element of the second 
row in data (remember that the first row is index 0). 
Then return elem.'''

def direct_index(data):
    elem = data[1][2]
    return elem
    pass

'''
Set slice1 equal to the specified first slice. 
Remember that NumPy uses a comma to separate slices along 
different dimensions.

Set slice2 equal to the specified second slice.

Return a tuple containing slice1 and slice2, in that order.
'''  

def slice_data(data):
    slice1 = data[:, 1:]
    slice2 = data[0:3, :-2]
    return slice1, slice2
    pass
    

'''
Set argmin_all equal to np.argmin with data as the only argument.

Set argmin1 equal to np.argmin with data as 
the first argument and the specified axis keyword argument.

Return a tuple containing argmin_all and argmin1, in that order
'''    

def argmin_data(data):
    argmin_all = np.argmin(data)
    argmin1 = np.argmin(data, axis=1)
    return argmin_all, argmin1
    pass
  
'''
Set argmax_neg1 equal to np.argmax with data as 
 the first argument and -1 as the axis keyword argument.
 Then return argmax_neg1.
'''
def argmax_data(data):
    argmax_neg1 = np.argmax(data, axis=-1)
    return argmax_neg1
    pass
    
      
  
  