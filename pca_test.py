from pca import *
# test
X = [[7, 2.1, 3.1],
     [1, 1.9, 3],
     [9, 1.9, 3]]

X = np.array(X)

m = mean2D(X)

# print(X)

# print(normalize2D(X))

# print(np.sum(X, axis=0))
# print(Xbar)
# print(eig(np.dot(X.T, X)))
result = PCA(X, 2)

print(result + m[:2])