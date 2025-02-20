import numpy

class Perceptron(object):
    """Perceptron Classifier.

    Parameters
    -----------
    eta: float
        Learning rate (0.0, 1.0)
    iter: int
        Passes over the training dataset

    Attributes
    -----------
    w_: 1d array
        Weights after fitting
    errors_: list
        Number of misclassification in every epoch
    """

    def __init__(self, eta=0.01, iter=10):
        self.eta = eta
        self.iter = iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        -----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features
        y: array-like, shape = [n_samples]
            Target values

        Returns
        -----------
        self : object

        """
        self.w_ = numpy.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return numpy.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label after unit step"""
        return numpy.where(self.net_input(X) >= 0.0, 1, -1)