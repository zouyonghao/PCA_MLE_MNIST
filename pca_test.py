from pca import *
# test
X = [[7, 2.1, 3.1],
     [1, 1.9, 3],
     [9, 1.9, 3]]

X = np.array(X)

# print(X)

# print(normalize2D(X))

# print(np.sum(X, axis=0))
# print(Xbar)
# print(eig(np.dot(X.T, X)))
print(PCA(X, 2))