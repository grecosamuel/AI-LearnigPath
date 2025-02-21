import numpy

class Adaline(object):
    """Adaline Classifire.
    
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
    def __init__(self, eta=0.01, iter=50):
        self.eta = eta
        self.iter = iter
    
    def fit(self, X, y):
        """
        Parameters
        -----------
        X: {array-like}. shape = [n_samples, n_features]
            Training vectors
            where n_samples is the number of samples and 
            n_features is the number of features. 
        y: array-like, shake = [n_samples]
            Target values
        
        Returns
        -----------
        self : object

        """
        # Initialize weights and costs
        self.w_ =  numpy.zeros(X.shape[1] + 1)
        self.cost_ = []

        # Iterate epochs
        for i in range(self.iter):
            output = self.net_input(X)
            
            # Check the error
            errors = (y - output)

            # Update weights
            self.w_[1:] += self.eta * X.T.dot(errors)

            # Update bias with SSE
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self 

    def net_input(self, X):
        """Calculate net input"""
        return numpy.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation""" 
        return self.net_input(X)
    
    def predict(self, X):
        """Return class label after unit step"""
        return numpy.where(self.activation(X) >= 0.0, 1, -1)