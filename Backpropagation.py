import math
import numpy as np

numberOfInputs = 2

w = np.array([0.01] * numberOfInputs)
b = 0.01
n = 0.5


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# x: input pattern, y: target label, z: actual label
def gradient(x, y, z):
    # return np.dot((sigmoid(z) - y) * sigmoid(z) * (1 - sigmoid(z)), x)
    return np.dot((z - y) * z * (1 - z), x)


def gradientBias(y, z):
    # return (sigmoid(z) - y) * sigmoid(z) * (1 - sigmoid(z))
    return (z - y) * z * (1 - z)


def gradientDescent(weight, x, y, z):
    return weight + (-n * gradient(x, y, z))


def gradientDescentBias(b, y, z):
    return b + (-n * gradientBias(y, z))


def perceptron(x, w, b):
    return sigmoid(np.dot(x, w) + b)


def updateWeights(w, x, y, z):
    return gradientDescent(w, x, y, z)


def updateBias(y, z, b):
    return gradientDescentBias(b, y, z)


# OR
x = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
y = [1, 1, 0, 1]

x = [[1, 1], [1, 0], [0, 0], [0, 1]]
y = [1, 1, 0, 1]

# XOR
# x = [[1, 1], [1, 0], [0, 0], [0, 1]]
# y = [0, 1, 0, 1]
#

# AND
# x = [[1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 1]]
# y = [1, 0, 0, 0, 1, 1]

for j in range(1000):
    for i in range(x.__len__()):
        z = perceptron(x[i], w, b)
        w = updateWeights(w, x[i], y[i], z)
        b = updateBias(y[i], z, b)
        # print(w, b)

print("results:")
for i in range(4):
    print(perceptron(x[i], w, b))
