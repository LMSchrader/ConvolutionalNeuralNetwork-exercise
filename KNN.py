import numpy as np
from sklearn import datasets
import statistics
import matplotlib.pyplot as plt

# import iris dataset
iris = datasets.load_iris()

pattern = iris.data[:, :]
label = iris.target


def euclideanDistance(pattern1, pattern2):
    a = 0
    for i in range(len(pattern1)):
        a += (pattern1[i] - pattern2[i]) ** 2
    return a


def knn(pattern, trainset_pattern, trainset_label, k):
    distList = np.zeros(len(trainset_pattern))
    for i in range(len(trainset_pattern)):
        distList[i] = euclideanDistance(pattern, trainset_pattern[i])

    sort_index = np.argsort(distList)

    knn = np.zeros(k)
    for i in range(k):
        knn[i] = trainset_label[sort_index[i]]

    return statistics.mode(knn)


# test
print(knn(pattern[0], pattern, label, 10))
print(label[0])
print(knn(pattern[100], pattern, label, 10))
print(label[100])

print(knn([1,2,3,4], pattern, label, 10))

# plot
plot_step = 0.02
plot_colors = "ryb"
n_classes = 3

# Plot the decision boundary
# for i in range(len(pattern)):

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    X = iris.data[:, pair]
    y = iris.target

    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    # plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    test = np.c_[xx.ravel(), yy.ravel()]
    Z = np.ones(len(test))
    for i in range(len(test)):
        Z[i] = knn(test[i], pattern, label, 10)

    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z)  # , cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    # Plot the training points
    #for j, color in zip(range(n_classes), plot_colors):
    #    idx = np.where(y == i)
    #    plt.scatter(X[idx, 0],
    #                X[idx, 1])  # , c=color, label=iris.target_names[i], cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.show()
