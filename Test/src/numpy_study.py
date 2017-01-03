import numpy as np
import time
from timeit import Timer
# arange([start,] stop[, step,], dtype=None)
# arange returns evenly spaced values within a given interval.
# a = np.arange(1, 10)
print(a)
# compare to range:
x = range(1, 10)
print(x)    # x is an iterator
print(list(x))
# some more arange examples:
# x = np.arange(10.4)
print(x)
# x = np.arange(0.5, 10.4, 0.8)
print(x)
# x = np.arange(0.5, 10.4, 0.8, int)
print(x)

# linspace(start, stop, num=50, endpoint=True, retstep=False)
# linspace returns an ndarray, consisting of 'num' equally spaced samples in the closed interval [start, stop]
# or the half-open interval [start, stop).
# 50 values between 1 and 10:
#print(np.linspace(1, 10))
# 7 values between 1 and 10:
#print(np.linspace(1, 10, 7))
# excluding the endpoint:
#print(np.linspace(1, 10, 7, endpoint=False))

#  If the optional parameter 'retstep' is set, the function will also return the value of the spacing
#  between adjacent values. So, the function will return a tuple ('samples', 'step'):
#samples, spacing = np.linspace(1, 10, retstep=True)
#print(spacing)
#samples, spacing = np.linspace(1, 10, 20, endpoint=True, retstep=True)
#print(spacing)
#samples, spacing = np.linspace(1, 10, 20, endpoint=False, retstep=True)
#print(spacing)


# One of the main advantages of NumPy is its advantage in time compared to standard Python.
# Let's look at the following functions:
size_of_vec = 1000
def pure_python_version():
    t1 = time.time()
    X = range(size_of_vec)
    Y = range(size_of_vec)
    Z = []
    for i in range(len(X)):
        Z.append(X[i] + Y[i])
    return time.time() - t1
def numpy_version():
    t1 = time.time()
    X = np.arange(size_of_vec)
    Y = np.arange(size_of_vec)
    Z = X + Y
    return time.time() - t1

t1 = pure_python_version()
t2 = numpy_version()
print(t1, t2)
print("Numpy is in this example " + str(t1/t2) + " faster!")


size_of_vec = 1000
def pure_python_version():
    X = range(size_of_vec)
    Y = range(size_of_vec)
    Z = []
    for i in range(len(X)):
        Z.append(X[i] + Y[i])
def numpy_version():
    X = np.arange(size_of_vec)
    Y = np.arange(size_of_vec)
    Z = X + Y
#timer_obj = Timer("x = x + 1", "x = 0")
timer_obj1 = Timer("pure_python_version()", "from __main__ import pure_python_version")
timer_obj2 = Timer("numpy_version()", "from __main__ import numpy_version")
print(timer_obj1.timeit(10))
print(timer_obj2.timeit(10))


# It's possible to create multidimensional arrays in numpy. Scalars are zero dimensional.
# In the following example, we will create the scalar 42. Applying the ndim method to our scalar,
#  we get the dimension of the array. We can also see that the type is a "numpy.ndarray" type.
x = np.array(42)
print("x: ", x)
print("The type of x: ", type(x))
print("The dimension of x:", np.ndim(x))

# We create array them by passing nested lists (or tuples) to the array method of numpy.
A = np.array([ [3.4, 8.7, 9.9],
               [1.1, -7.8, -0.7],
               [4.1, 12.3, 4.8]])
print(A)
print(A.ndim)
B = np.array([ [[111, 112], [121, 122]],
               [[211, 212], [221, 222]],
               [[311, 312], [321, 322]] ])
print(B)
print(B.ndim)
print(B.shape)


F = np.array([1, 1, 2, 3, 5, 8, 13, 21])
# print the first element of F, i.e. the element with the index 0
print(F[0])
# print the last element of F
print(F[-1])
B = np.array([ [[111, 112], [121, 122]],
               [[211, 212], [221, 222]],
               [[311, 312], [321, 322]] ])
print(B[0][1][0])

A = np.array([ [3.4, 8.7, 9.9],
               [1.1, -7.8, -0.7],
               [4.1, 12.3, 4.8]])
print(A[1][0])
print(A[1, 0])  # This is more efficient

S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(S[2:5])
print(S[:4])
print(S[6:])
print(S[:])

A = np.array([
[11, 12, 13, 14, 15],
[21, 22, 23, 24, 25],
[31, 32, 33, 34, 35],
[41, 42, 43, 44, 45],
[51, 52, 53, 54, 55]])
print(A[:3, 2:])

# The following two examples use the third parameter "step".
# The reshape function is used to construct the two-dimensional array.
X = np.arange(28).reshape(4,7)
print(X)
print(X[::2, ::3])
print(X[::, ::3])

# Attention: Whereas slicings on lists and tuples create new objects, a slicing operation on an array
# creates a view on the original array. So we get an another possibility to access the array,
# or better a part of the array. From this follows that if we modify a view,
# the original array will be modified as well.
A = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
S = A[2:6]
S[0] = 22
S[1] = 23
print(A)

# With list we get a copy
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
lst2 = lst[2:6]
lst2[0] = 22
lst2[1] = 23
print(lst)

# If you want to check, if two array names share the same memory block, you can use the function np.may_share_memory.
np.may_share_memory(A, S)

# The following code shows a case, in which the use of may_share_memory is quite useful:
A = np.arange(12)
B = A.reshape(3, 4)
A[0] = 42
print(B)

# We can see that A and B share the memory in some way. The array attribute "data" is an object pointer
# to the start of an array's data. Looking at the data attribute returns something surprising:
print(A.data)
print(B.data)
print(A.data == B.data)

# Let's check now on equality of the arrays:
print(A == B)

# Which makes sense, because they are different arrays concerning their structure:
print(A)
print(B)

# But we saw that if we change an element of one array the other one is changed as well.
# This fact is reflected by may_share_memory:
np.may_share_memory(A, B)


E = np.ones((2, 3))
print(E)
F = np.ones((3, 4), dtype=int)
print(F)
# What we have said about the method ones() is valid for the method zeros() analogously,
# as we can see in the following example:


Z = np.zeros((2, 4))
print(Z)
# There is another interesting way to create an array with Ones or with Zeros, if it has to have the same shape
#  as another existing array 'a'. Numpy supplies for this purpose the methods ones_like(a) and zeros_like(a).
x = np.array([2, 5, 18, 14, 4])
E = np.ones_like(x)
print(E)
Z = np.zeros_like(x)
print(Z)

x = np.array([[42, 22, 12], [44, 53, 66]], order='F')
y = x.copy()
x[0, 0] = 1001
print(x)
print(y)
print(x.flags['C_CONTIGUOUS'])
print(y.flags['C_CONTIGUOUS'])

I = np.identity(4)

# Another way to create identity arrays provides the function eye. It returns a 2-D array with
# ones on the diagonal and zeros elsewhere.
# eye(N, M=None, k=0, dtype=float)
# Parameter	Meaning
# N	An integer number defining the rows of the output array.
# M	An optional integer for setting the number of columns in the output. If it is None, it defaults to 'N'.
# k	Defining the position of the diagonal. The default is 0. 0 refers to the main diagonal. A positive value refers
# to an upper diagonal, and a negative value to a lower diagonal.
# dtype	Optional data-type of the returned array.
# eye returns an ndarray of shape (N,M). All elements of this array are equal to zero, except for the 'k'-th diagonal,
#  whose values are equal to one.
np.eye(5, 8, k=1, dtype=int)

#  Exercises:
#  1) Create an arbitrary one dimensional array called "v".
#  2) Create a new array which consists of the odd indices of previously created array "v".
#  3) Create a new array in backwards ordering from v.
#  4) What will be the output of the following code:
#   a = np.array([1, 2, 3, 4, 5])
#   b = a[1:4]
#   b[0] = 200
#   print(a[1])
#  5) Create a two dimensional array called "m".
#  6) Create a new array from m, in which the elements of each row are in reverse order.
#  7) Another one, where the rows are in reverse order.
#  8) Create an array from m, where columns and rows are in reverse order.
#  9) Cut of the first and last row and the first and last column.

#  Solutions
#  1)
#  import numpy as np
a = np.array([3, 8, 12, 18, 7, 11, 30])
#  2)
odd_elements = a[1::2]
#  3)
reverse_order = a[::-1]
#  4) The output will be 200, because slices are views in numpy and not copies.
#  5)
m = np.array([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]])
#  6)
m[::, ::-1]
#  7)
m[::-1]
#  8)
m[::-1, ::-1]
#  9)
m[1:-1, 1:-1]




