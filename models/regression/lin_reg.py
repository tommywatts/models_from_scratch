import numpy as np
from itertools import combinations_with_replacement, chain

from models.regression.base import LinearBase


class Linreg(LinearBase):
    
    """ Linear Regression """
    
    def __init__(self, learning_rate = 0.01, n_iter = 1000):
        super().__init__(learning_rate, n_iter)
        self.coefs = None
        self.intercept = None
        self.fitted = False 

    def fit(self, X, y, optimization='normal'):

        self.m, self.n = X.shape
                 
        # add 1s for bias 
        xb = np.hstack((np.ones((self.m, 1)), X))

        # initiate coefs
        self.theta = np.ones((xb.shape[1], 1))
        
        if optimization=='normal':
            if self.m < 50000:
                self.theta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
            else:
                optimization = 'gradient_descent'
        
        if optimization=='gradient_descent':
            
            super().fit(xb, y)
        
        if optimization=='stochastic':

            def learning_schedule(t, t0=5):
                return t0 / (t + self.n_iter)
            
            for step in range(self.n_iter):
                
                for i in range(self.m): 
                    
                    # random selection from input
                    random_index = np.random.randint(self.m)
                    xi = xb[random_index:random_index + 1]
                    yi = y[random_index:random_index + 1]
                    
                    # compute error
                    error = self._error(xi, yi, self.theta)
                    
                    # update gradients
                    gradients =  2 * xi.T.dot(error)
                
                    # get learning rate at each iteration from the schedule
                    eta = learning_schedule(step * self.m + i)
                    
                    self.theta -= eta * gradients
            
        # fit attributes
        self.coefs = self.theta[1:].flatten()
        self.intercept = self.theta[0].flatten()
        self.fitted = True
        return self
        

        

class PolyReg(LinearBase):

    """Linear Regression with polynomial transformation"""

    def __init__(self, learning_rate = 0.01, n_iter = 1000, degree=1):
        super().__init__(learning_rate, n_iter)
        self.degree = degree
        self.coefs = None
        self.intercept = None
        self.fitted = False 

    def _polynomial_transformation(X, degree):
                
        feature_combinations = list(chain.from_iterable(combinations_with_replacement(range(self.n), i)
                                        for i in range(0, degree + 1)))

        Xn = np.zeros((self.m, len(feature_combinations)))

        for i, comb in enumerate(feature_combinations):
            Xn[:, i] = np.prod(X[:, comb], axis=1)

        return Xn

    def fit(X, y, optimization='normal'):

        self.m, self.n = X.shape

        xb = self._polynomial_transformation(X, self.degree)

        self.theta = np.ones((xb.shape[1], 1))

        if optimization=='normal':
            if self.m < 50000:
                self.theta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
            else:
                optimization = 'gradient_descent'
        
        if optimization=='gradient_descent':
            
            super().fit(xb, y)

        self.coefs = self.theta[1:].flatten()
        self.intercept = self.theta[0].flatten()
        self.fitted = True
        return self


        