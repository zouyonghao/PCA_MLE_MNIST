import numpy as np

# X must be a 2D array
def mean2D(X):
    mean = np.zeros_like(X[0], dtype=float)
    sum = np.zeros_like(X[0], dtype=float)
    count = len(X)
    for i in range(len(X)):
        # for j in range(len(X[0])):
        #     sum[j] += X[i][j]
        sum = sum + X[i]
    for i in range(len(X[0])):
        mean[i] = sum[i] / count
    return mean

# X must be a 2D array
# convert X to zero mean
def normalize2D(X):
    newX = np.zeros(np.array(X).shape)
    m = mean2D(X)
    # print(m)
    for i in range(len(X)):
        # for j in range(len(X[0])):
        #     newX[i][j] = X[i][j] - m[j]
        newX[i] = X[i] - m
    return newX


def eig(X):
    # print(X.shape[0])
    eigenValues, eigenVectors = np.linalg.eig(X)
    idx = eigenValues.argsort()[::-1]
    # print(idx)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[idx]
    return (eigenValues, eigenVectors)

def PCA(X, dimensions):
    X = normalize2D(X)
    eigenValues, eigenVectors = eig(np.dot(X.T, X) / X.shape[0])
    # print(eigenVectors[:, :dimensions])
    return np.dot(X, eigenVectors[:, :dimensions])

# test
# X = [[7, 2.1, 3.1],
#      [1, 1.9, 3],
#      [9, 1.9, 3]]

# X = np.array(X)

# print(X)

# print(normalize2D(X))

# print(np.sum(X, axis=0))
# print(Xbar)
# print(eig(np.dot(X.T, X)))
# print(PCA(X, 2))

