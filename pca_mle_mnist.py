import numpy as np
import struct
import matplotlib.pyplot as plt
import pylab
import pca
import mle
import tools

train_data = tools.get_train_data()

# print(train_data[0])
# print(len(train_data[0]))
print(len(train_data))

# plt.imshow(im[0], cmap='gray')
# pylab.show()

train_label = tools.get_train_label()

print(len(train_label))

dimension = 90

train_data, eigenVectors = pca.PCA(train_data, dimension)

print(np.array(train_data).shape)

# 10 classes
classified_data = []
for i in range(10):
    classified_data.append([])

for i in range(len(train_label)):
    # print(train_label[i][0])
    classified_data[train_label[i][0]].append(train_data[i])

train_data_mean = []
train_data_cov = []
for i in classified_data:
    # i是每行表示一个数据
    mean = mle.mean2D(i)
    train_data_mean.append(mean)
    train_data_cov.append(mle.cov(i, mean))

raw_test_data = tools.get_test_data()

# raw_test_data = pca.normalize2D(raw_test_data)
test_data = np.matmul(raw_test_data, eigenVectors)
test_label = tools.get_test_label()

correct = 0
count = 0

for i in range(len(test_data)):
    count += 1
    predict = np.zeros(len(train_data_mean), )
    for j in range(10):
        predict[j] = mle.gaussian(
            test_data[i], train_data_mean[j], train_data_cov[j])

    label = 0
    max = 0
    for k in range(10):
        if predict[k] > max:
            max = predict[k]
            label = k
    if (label == test_label[i][0]):
        correct += 1
        print(correct)
        print(count)

print(correct / len(test_data))
