import numpy
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

epochs = 40 # Set epochs
learningRate = 0.01 # Set Learning Rate
testSize = 0.2 # Set percentage of dataset splitting for test prediction

# Load IRIS Dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Divide dataset to use 30% of data for training and the rest for prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=0)

# Use StandardScaler for standardization
stdScaler = StandardScaler()
stdScaler.fit(X_train)
X_train_std = stdScaler.transform(X_train)
X_test_std = stdScaler.transform(X_test)

# Define Perceptron
perceptron = Perceptron(max_iter=epochs, eta0=learningRate, random_state=0)
perceptron.fit(X_train_std, y_train)

pred = perceptron.predict(X_test_std)
errors = (y_test != pred).sum()

print(f"Epochs: {epochs}, Learning rate: {learningRate}")
print(f"Length of training data: {len(X_train_std)}")
print(f"Length of test data: {len(X_test_std)}" )
print("Accuracy: %.2f"  % accuracy_score(y_test, pred))
print(f"Errors: {errors}")