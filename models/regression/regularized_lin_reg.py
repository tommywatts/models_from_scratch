import numpy as np

from models.regression.base import LinearBase

class RidgeReg(LinearBase):
    
    """Ridge Regression"""
    
    def __init__(self, learning_rate = 0.01, n_iter = 1000, alpha=1):
        super().__init__(learning_rate, n_iter)
        self.coefs = None
        self.intercept = None
        self.fitted = False 
        self.alpha = alpha

    def fit(self, X, y, optimization='normal'):
        self.m, self.n = X.shape
        
        # add 1s for bias
        xb = np.hstack((np.ones((self.m, 1)), X))
        
        # initiate coefs
        self.theta = np.ones((xb.shape[1], 1))
        
        # closed form
        if optimization=='normal':
            if self.m < 50000:

                # identity matrix
                idm = np.eye(len(self.theta))
                idm[0][0] = 0
                A = self.alpha * idm  
    
                self.theta = np.linalg.inv(xb.T.dot(xb) + A).dot(xb.T).dot(y)
            else:
                optimization = 'gradient_descent'
                
        if optimization=='gradient_descent':
            for step in range(self.n_iter):
                
                # compute error
                error = self._error(xb, y, self.theta)

                 # update gradients 
                gradients = 2 / self.m * xb.T.dot(error) + (2 * self.alpha * self.theta)

                # update coefficients            
                theta -= self.learning_rate * gradients
        
        self.coefs = self.theta[1:].flatten()
        self.intercept = self.theta[0].flatten()
        self.fitted = True
        return self
   


class LassoReg(LinearBase):
    
    """LASSO Regression using coordinate gradient descent"""
    
    def __init__(self, n_iter=1000, alpha=1):
        super().__init__(learning_rate=None, n_iter=1000)
        self.alpha = alpha

    def fit(self, X, y):
        self.m, self.n = X.shape
        
        # add 1s for intercept
        xb = np.hstack((np.ones((self.m, 1)), X))
            
        # initialise theta    
        theta = np.zeros(xb.shape[1])

        # set intercept
        theta[0] = np.sum(y - np.dot(xb[:, 1:], theta[1:])) / self.m
        
        for step in range(self.n_iter):
            
            for j in range(1, self.n):
                
                # temporary theta
                theta_ = theta
                theta_[j] = 0.0
                
                # residuals
                err = y - np.dot(xb, theta_)
                
                # input to thresholding
                h = np.dot(xb[:, j], err)
                lambda_ = self.alpha * self.m

                # update coefficient
                theta[j] = self._soft_thresholding(h, lambda_) / (xb[:, j] ** 2).sum()

                # set intercept
                theta[0] = np.sum(y - np.dot(xb[:, 1:], theta[1:])) / self.m

    
        self.intercept = theta[0].flatten()
        self.coefs = theta[1:].flatten()
        self.fitted = True
        return self

    @staticmethod
    def _soft_thresholding(x, lambda_):
        if x > 0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0 and lambda_ < abs(x):
            return x + lambda_
        return 0
   