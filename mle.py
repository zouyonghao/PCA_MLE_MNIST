#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'mle'

__author__ = 'Yong-Hao Zou'

import numpy as np
import math

# X must be a 2D array


def mean2D(X):
    # X = X.T
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


def cov(X, mean):
    X = np.array(X)
    print(X.shape)
    # tmp = X
    tmp = X - mean
    # tmp = tmp.T
    # print(tmp)
    # print(np.dot(tmp, tmp.T))
    return np.matmul(tmp.T, tmp)
    # result = np.matmul(tmp.T, tmp) / (len(tmp) - 1)
    # a = np.zeros(result.shape)
    # np.fill_diagonal(a, 0.1)

    # return np.matmul(result, a)


def gaussian(X, mean, cov):
    tmp = X - mean
    # print(np.linalg.det(var))
    # print(tmp.T.dot(np.linalg.inv(var)).dot(tmp))
    # print(1 / math.sqrt(np.linalg.det(var)))
    # print(math.exp(- 0.5 * tmp.T.dot(np.linalg.inv(var)).dot(tmp)))
    sign, logdet = np.linalg.slogdet(cov)
    # print(np.linalg.det(cov))
    
    return (1 / math.sqrt(sign * np.log(logdet))) * math.exp(- 0.5 * tmp.dot(np.linalg.inv(cov)).dot(tmp.T))
    # return math.fabs(tmp.sum())
    # return (1 / math.sqrt(np.linalg.det(cov))) * math.exp(- 0.5 * tmp.dot(np.linalg.inv(cov)).dot(tmp.T))


if __name__ == "__main__":
    X = np.array([
        [1, 8, 1],
        [4, 5, 8]
    ])
    mean = mean2D(X)
    print(mean)
    var = cov(X, mean)
    print("var")
    print(var)

    print("np.cov")
    print(np.cov(X.T))

    # Y = np.array([1, 3, 9])
    print(gaussian(X, mean, var))
