import numpy as np

from models.regression.base import LinearBase

class LogReg(LinearBase):

    def __init__(self, learning_rate = 0.01, n_iter = 1000):
        super().__init__(learning_rate, n_iter)
        self.coefs = None
        self.intercept = None
        self.fitted = False 

    def fit(self, X, y):
        
        self.m, self.n = X.shape
                 
        # add 1s for bias 
        xb = np.hstack((np.ones((self.m, 1)), X))

        # initiate coefs
        self.theta = np.ones((xb.shape[1], 1))
        
        for step in range(self.n_iter):
            
  
            error = self.sigmoid(xb.dot(self.theta)) - y.reshape(-1,1)

            # update gradients 
            gradients = 2 / self.m * xb.T.dot(error)
        
            # update theta 
            self.theta -= self.learning_rate * gradients
            
        
        self.coefs = self.theta[1:].flatten()
        self.intercept = self.theta[0].flatten()
        self.fitted = True
        return self
    
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z)) 

    def predict_proba(self, X):
        return self.sigmoid(super().predict(X))

    def predict(self, X, threshold = .5):
        return np.where(self.predict_proba(X) > threshold, 1, 0)