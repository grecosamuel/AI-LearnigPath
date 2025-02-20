import pandas, numpy
from perceptron import Perceptron
import matplotlib.pyplot as plt

IRIS_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

dataset = pandas.read_csv(IRIS_DATASET_URL, header=None)
dataset.tail()

# Create 1D array with first 100 record from IRIS Dataset, with iris label name
y = dataset.iloc[0:100, 4].values

# Create 1D array from array y and assign -1 to 'Iris-setosa' value and 1 to 'Iris-versicolor' value
y = numpy.where(y == 'Iris-setosa', -1, 1)

# Create 2D array from IRIS Dataset with Sepal Length and Petal Lenght in the first 2 columns of dataset
X = dataset.iloc[0:100, [0,2]].values


# Show graph 
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')

plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')

plt.show()

# Implement Perceptron on IRIS Dataset
perceptronAlg = Perceptron(eta=0.1)
perceptronAlg.fit(X, y)

# Show graph result
plt.plot(range(1, len(perceptronAlg.errors_) + 1), perceptronAlg.errors_, marker='o')
plt.xlabel('Epoch')
plt.ylabel('N misclassification')
plt.show()