#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'mle'

__author__ = 'Yong-Hao Zou'

import numpy as np
import math

# X must be a 2D array


def mean2D(X):
    X = X.T
    mean = np.zeros_like(X[0], dtype=float)
    sum = np.zeros_like(X[0], dtype=float)
    count = len(X)
    for i in range(len(X)):
        # for j in range(len(X[0])):
        #     sum[j] += X[i][j]
        sum = sum + X[i]
    for i in range(len(X[0])):
        mean[i] = sum[i] / count
    return mean.T


def var2D(X, mean):
    X = np.array(X)
    print(X.shape)
    tmp = X.T - mean
    tmp = tmp.T
    # print(tmp)
    # print(np.dot(tmp, tmp.T))
    return np.dot(tmp, tmp.T)/(X.shape[0] - 1)


def gaussian(X, mean, var):
    tmp = X - mean
    # print(np.linalg.det(var))
    # print(tmp.T.dot(np.linalg.inv(var)).dot(tmp))
    # print(1 / math.sqrt(np.linalg.det(var)))
    # print(math.exp(- 0.5 * tmp.T.dot(np.linalg.inv(var)).dot(tmp)))
    # return (1 / math.sqrt(np.linalg.det(var))) * math.exp(- 0.5 * tmp.T.dot(np.linalg.inv(var)).dot(tmp))
    return math.fabs(tmp.sum())


if __name__ == "__main__":
    X = np.array([
        [1, 8, 1],
        [4, 5, 1],
        [1, 4, 9]
    ])
    mean = mean2D(X)
    print(mean)
    var = var2D(X, mean)
    print(var)

    Y = np.array([1, 3, 9])
    print(gaussian(Y, mean, var))
