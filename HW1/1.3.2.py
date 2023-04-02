import numpy as np


n = 8 
B = np.random.rand(n, n)
A = np.dot(B, B.transpose())
x0 = np.random.rand(n,1)

print(x0)
print(A)
x = [x0] 
count = 0
size = 60
while count <= size:
    y = np.dot(A,x[-1])
    print(x[count])
    x.append(y/np.linalg.norm(y, ord=None, axis=None, keepdims=False))
    count = count + 1

lam = np.linalg.eig(A)
index = np.argmax(lam[0])
lamda_max = np.real(lam[0][index])
vector = lam[1][:,index]
vector_final = np.transpose((np.real(vector)))
print(lamda_max, vector_final)
print(vector_final-x[-1])