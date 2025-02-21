import pandas, numpy
from adaline import Adaline
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
#plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')

#plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

#plt.xlabel('sepal length')
#plt.ylabel('petal length')
#plt.legend(loc='upper left')

#plt.show()

# Implement Adaline GD on IRIS Dataset

## Train first model
ada = Adaline(eta=0.01, iter=10)
ada.fit(X, y)

fig, ax = plt.subplots(2, 2, figsize=(8, 4))
ax[0][0].plot(range(1, len(ada.cost_) + 1), numpy.log10(ada.cost_), marker='o')
ax[0][0].set_xlabel('Epochs')
ax[0][0].set_ylabel('Sum SSE')
ax[0][0].set_title('ETA: 0.01')

## Train second model
ada2 = Adaline(eta=0.0001, iter=10)
ada2.fit(X, y)

ax[0][1].plot(range(1, len(ada2.cost_) + 1), numpy.log10(ada2.cost_), marker='o')
ax[0][1].set_xlabel('Epochs')
ax[0][1].set_ylabel('Sum SSE')
ax[0][1].set_title('ETA: 0.0001')

# Standardization
X_std = numpy.copy(X)
X_std[: , 0] = (X_std[: , 0] - X_std[: , 0].mean()) / X_std[: , 0].std()
X_std[: , 1] = (X_std[: , 1] - X_std[: , 1].mean()) / X_std[: , 1].std()

# Train third model
ada3 = Adaline(eta=0.01, iter=10)
ada3.fit(X_std, y)
ax[1][0].plot(range(1, len(ada3.cost_) + 1), ada3.cost_, marker='o')
ax[1][0].set_xlabel('Epochs')
ax[1][0].set_ylabel('SSE')
ax[1][0].set_title('Standardized dataset')

plt.show()