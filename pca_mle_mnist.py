import numpy as np
import struct
import matplotlib.pyplot as plt
import pylab
from pca import *

train_file = open("train-images.idx3-ubyte", "rb")

buf = train_file.read()
index = struct.calcsize('>IIII')
train_data = []
while index < len(buf):
    tmp = struct.unpack_from('>784B', buf, index)
    # im.append(np.reshape(tmp, (28, 28)))
    train_data.append(tmp)
    index += struct.calcsize('>784B')

# print(train_data[0])
# print(len(train_data[0]))
print(len(train_data))

# plt.imshow(im[0], cmap='gray')
# pylab.show()

train_label_file = open("train-labels.idx1-ubyte", "rb")
buf = train_label_file.read()
index = struct.calcsize('>II')
train_label = []

while index < len(buf):
    tmp = struct.unpack_from('>B', buf, index)
    train_label.append(tmp)
    # print(tmp)
    index += struct.calcsize('>B')

print(len(train_label))

train_data = PCA(train_data, 100)

print(np.array(train_data).shape)