import pandas, numpy
from adalineSGD import AdalineSGD
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

# Standardization
X_std = numpy.copy(X)
X_std[: , 0] = (X_std[: , 0] - X_std[: , 0].mean()) / X_std[: , 0].std()
X_std[: , 1] = (X_std[: , 1] - X_std[: , 1].mean()) / X_std[: , 1].std()

# Train model
ada = AdalineSGD(iter=10, random_state=1)
ada.fit(X_std, y)

# Show graph
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Average cost")
plt.show()