class LinearBase(object):
    
    """Base class for linear models"""
    
    def __init__(self, learning_rate, n_iter):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit using gradient descent.
        """
        for i in range(self.n_iter):

            error = self._error(X, y, self.theta)

            # update gradients 
            gradients = 2 / self.m * X.T.dot(error)
        
            # update theta 
            self.theta -= self.learning_rate * gradients
            
    def _error(self, X, y, theta):
        return X.dot(theta) - y.reshape(-1,1)

    def predict(self, X):
        xm, xn = X.shape
        
        if not self.fitted:
            raise Exception(f"{__class__.__name__} is not yet fitted.")
            
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
            
        if xn != self.n:
            raise Exception(f"Input data shape must be equal to fit data shape {self.n}")
            
        if xn == 1:
            return X.dot(self.coefs) + self.intercept
        
        return X.dot(self.coefs)
    
    def __repr__(self):
        if self.fitted:
            return f"coefficients: {self.coefs}, \n\n intercept: {self.intercept}"
        return self.__class__.__name__