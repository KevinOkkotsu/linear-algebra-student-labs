import numpy as np

'''
# constructing a matrix
A = np.array([[10, 1, 0, 9],[12.4, 6, 1, 0], [0, 6, 0, 8], [1, 3.14, 1, 0]])

#print(A, A.shape)

identity = np.eye(4)        # argument in np.eye() specifies matrix dimension
#print(identity)

zeros = np.zeros((4,4))     # needs two brackets for argument
#print(zeros)

B = np.empty((4,4))         # needs two brackets for argument
#print(B)
'''

'''
# Scalar multiplication
print(5 * A)
# Identity + A
x = identity + A
print(x)
# @ operator to perform matrix multiplication
y = zeros @ A
print(y)
# return the transpose of A
print(A.T)
'''

'''
# row vector
a = np.array([1.0, 2.0, 3.0])
# column vector
b = np.array([[1.0, 2.0, 3.0],[3,5,9]])

# computing scalar products and printing entry type
c = np.array([-1.0, 0.0, -1.0])
print(np.dot(a, c), c.dtype)
# we can ensure we are using a floating point type on entries
d = np.array([1.0,2.0,3.0], dtype=float) # float type
d = np.array([1.0,2.0,3.0], dtype=np.double) # double precision
d = d.astype(float) # converts the existing matrix d entries to float type

# Two numbers that should be mathematically equal
'''

'''
a = 0.1 + 0.2
b = 0.3

print(f"a = {a}")      # a = 0.30000000000000004
print(f"b = {b}")      # b = 0.3
print(f"a == b: {a == b}")  # False!


# solution: if euclidean distance is between tolerance
tol = 1.08e-8
if np.linalg.norm(b-c) < tol:
    print("they are equal!")
else:
    print("they are not equal!")
'''

# Exercise 2
A = np.array([[1,2],[3,4]])

def determinant(A):
    n, m = A.shape
    assert n == m, f"A is not square! {A.shape=}"

    det = 0
    if n == 2:
        det = (A[0,0]*A[1,1]) - (A[0,1]*A[1,0]) 
        return  det

    elif n == 3:
        det =(A[0,0]*(A[1,1]*A[2,2] - A[1,2]*A[2,1])) - (A[0,1]*((A[1,0]*A[2,2])-A[1,2]*A[2,0])) + (A[0,2]*(A[1,0]*A[2,1]-A[1,1]*A[2,0]))
        return det

A = np.array([[1,2],[3,4]])

B = np.array([[1, 2, 3],
              [0, 4, 5], 
              [1, 0, 6]])
print(determinant(B))