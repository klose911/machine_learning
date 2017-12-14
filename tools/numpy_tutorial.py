import numpy as np  
a = np.array([0,1,2,3,4,5])
a.ndim # 1 
a.shape # (6,) 

b = a.reshape(3,2)  
b 
# array([[0, 1],
# [2, 3],
# [4, 5]])
b.ndim # 2 
b.shape # (3,2) 

b[1][0]=77 
a # array([ 0,  1, 77,  3,  4,  5]) 

c = a.reshape((3,2)).copy() 
c[0][0] = -99 
a # array([ 0,  1, 77,  3,  4,  5]) 

a * 2 # array([  0,   2, 154,   6,   8,  10]) 
a ** 2 # array([   0,    1, 5929,    9,   16,   25]) 

# use arrays themselves as indices 
a[np.array([2,3,4])]  
#array([77,  3,  4])


a>4 # array([False, False,  True, False, False,  True], dtype=bool)
a[a>4] # array([77,  5]) 

a[a>4] = 4 # array([0, 1, 4, 3, 4, 4])
a.clip(0,3)  # array([0, 1, 3, 3, 3, 3])

# NAN : invalid values 
c = np.array([1, 2, np.NAN, 3, 4]) 
np.isnan(c) # array([False, False,  True, False, False], dtype=bool)
c[~np.isnan(c)] # array([ 1.,  2.,  3.,  4.])
np.mean(c[~np.isnan(c)]) # 2.5 

import timeit
normal_py_sec = timeit.timeit('sum(x*x for x in xrange(1000))',number=10000)
naive_np_sec = timeit.timeit('sum(na*na)',
                             setup="import numpy as np; na=np.arange(1000)", number=10000)
good_np_sec = timeit.timeit('na.dot(na)',
                            setup="import numpy as np; na=np.arange(1000)", number=10000) 

print("Normal Python: %f sec"%normal_py_sec)
print("Naive NumPy: %f sec"%naive_np_sec)
print("Good NumPy: %f sec"%good_np_sec) 

# NumPy arrays always have only one datatype!!!
np.array([1, "stringy"]) # array(['1', 'stringy'], dtype='|S11')
