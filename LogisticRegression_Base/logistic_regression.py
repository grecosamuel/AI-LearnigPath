import numpy
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

C = 1000.0 # Regolarization value parameter
testSize = 0.3 # Set percentage of dataset splitting for test prediction

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

# Define and train model
logisticRegression = LogisticRegression(C=C, random_state=0)
logisticRegression.fit(X_train_std, y_train)

# Predict probability of classification
prob_prediction = logisticRegression.predict_proba(X_test_std)

# Apply classification
prediction = logisticRegression.predict(X_test_std)

# Measure accuracy score and errors
score = accuracy_score(y_test, prediction)
errors = (y_test != prediction).sum()

# Show results
for index, target in enumerate(X_test_std):
    print("Probability score: %.2f" % (numpy.max(prob_prediction[index])))
    print(f"Predicted value: {prediction[index]}")
    print(f"Target value: {y_test[index]}\n\n")

# Show general metrics
print(f"Length of training data: {len(X_train_std)}")
print(f"Length of test data: {len(X_test_std)}" )
print("Accuracy: %.2f"  % score)
print(f"Errors: {errors}")