from sklearn.datasets import make_regression, make_blobs
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor, LogisticRegression
from sklearn.model_selection import train_test_split
import unittest
import numpy as np
import warnings

from context import Linreg, RidgeReg, LassoReg, LogReg


class TestRegression(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=PendingDeprecationWarning)

    def test_linear_regression_norm(self):
        X, y = make_regression(n_samples=20000, n_features=5, noise=5)

        linreg = Linreg().fit(X, y)

        sk_linreg = LinearRegression().fit(X, y)

        np.testing.assert_array_almost_equal(linreg.coefs, sk_linreg.coef_, 1)
        np.testing.assert_array_almost_equal(linreg.intercept, sk_linreg.intercept_)

        pred_data = np.random.uniform(0, 2, 5).reshape(1, -1)

        np.testing.assert_array_almost_equal(linreg.predict(pred_data), 
                                             sk_linreg.predict(pred_data), 1)

    def test_linear_regression_gradient_descent(self):
        X, y = make_regression(n_samples=100000, n_features=5, noise=5)

        linreg = Linreg(n_iter=2000).fit(X, y)

        sk_linreg = LinearRegression().fit(X, y)

        np.testing.assert_array_almost_equal(linreg.coefs, sk_linreg.coef_, 1)
        np.testing.assert_array_almost_equal(linreg.intercept, sk_linreg.intercept_)

        pred_data = np.random.uniform(0, 2, 5).reshape(1, -1)

        np.testing.assert_array_almost_equal(linreg.predict(pred_data), 
                                             sk_linreg.predict(pred_data), 1)
    
    def test_linear_regression_stochastic(self):
        X, y = make_regression(n_samples=20000, n_features=5, noise=5)

        linreg = Linreg(n_iter=50).fit(X, y, optimization='stochastic')

        sk_SGD = SGDRegressor(penalty=None, eta0=0.1).fit(X, y)

        np.testing.assert_array_almost_equal(linreg.coefs, sk_SGD.coef_, 0)
        np.testing.assert_array_almost_equal(linreg.intercept, sk_SGD.intercept_, 0)

        pred_data = np.random.uniform(0, 2, 5).reshape(1, -1)

        np.testing.assert_array_almost_equal(linreg.predict(pred_data), 
                                             sk_SGD.predict(pred_data), 0)


    def test_poly(self):
        pass

    def test_ridge(self):
        X, y = make_regression(20000, 20, noise=20, n_informative=10)

        ridge = RidgeReg(alpha=0.5).fit(X, y)
        sk_ridge = Ridge(alpha=0.5).fit(X, y)

        np.testing.assert_array_almost_equal(ridge.coefs, sk_ridge.coef_, 1)
        np.testing.assert_array_almost_equal(ridge.intercept, sk_ridge.intercept_)

        pred_data = np.random.uniform(0, 2, 20).reshape(1, -1)

        np.testing.assert_array_almost_equal(ridge.predict(pred_data), 
                                             sk_ridge.predict(pred_data), 0)


    def test_lasso(self):
        X, y = make_regression(1000, 20, noise=20, n_informative=5)
        X, Xp, y, yp = train_test_split(X, y, test_size=0.01, random_state=42)

        lasso = LassoReg(alpha=.5).fit(X, y)
        sk_lasso = Lasso(alpha=.5).fit(X, y)
        
        np.testing.assert_array_almost_equal(lasso.coefs, sk_lasso.coef_, 0)
        np.testing.assert_array_almost_equal(lasso.intercept, sk_lasso.intercept_, 0)

        np.testing.assert_array_almost_equal(lasso.predict(Xp), 
                                      sk_lasso.predict(Xp), 0)



    def test_logit(self):
        X, y = make_blobs(centers=2, n_samples=500)
        X, Xp, y, yp = train_test_split(X, y, test_size=0.01, random_state=42)

        logit = LogReg().fit(X, y)

        sk_logit = LogisticRegression().fit(X, y)

        np.testing.assert_array_equal(logit.predict(Xp), 
                          sk_logit.predict(Xp))


if __name__ == '__main__':
    unittest.main()