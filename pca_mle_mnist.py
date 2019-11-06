import numpy as np
import struct
import matplotlib.pyplot as plt
import pylab

train_data = open("train-images.idx3-ubyte", "rb")

buf = train_data.read()
index = struct.calcsize('>IIII')
im = []
while index < len(buf):
    tmp = struct.unpack_from('>784B', buf, index)
    im.append(np.reshape(tmp, (28, 28)))
    index += struct.calcsize('>784B')

print(len(im))

plt.imshow(im[0], cmap='gray')
pylab.show()