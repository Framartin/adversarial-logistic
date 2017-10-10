import statsmodels.api as sm
from sklearn import linear_model
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import special, stats


class AdversarialLogistic(object):
    """docstring for AdversarialLogit"""

    def __init__(self, model, X_train=None, lower_bound=float('-inf'), upper_bound=float('inf')):
        super(AdversarialLogit, self).__init__()
        # check if model is supported
        # check that not multinomial logit
        # check that we use the bernoouilli Logit, not the Binomial
        self.model = model
        module = getattr(model, '__module__')
        self.need_add_constant = False
        if(module == 'sklearn.linear_model.logistic'):
            self.module = 'sklearn'
            if model.get_params()['fit_intercept']:
                self.beta_hat_minus0 = model.coef_
                self.beta_hat = np.insert(model.coef_, 0, model.intercept_)
                self.need_add_constant = True
            else:
                self.beta_hat_minus0 = self.beta_hat = model.coef_.squeeze()
        elif(self.module == 'statsmodels.genmod.generalized_linear_model'):
            self.module = 'statsmodels'
            assert(X_train is not None)
            idx_beta0 = detect_constant(X_train)
            self.beta_hat = model.params
            self.beta_hat_minus0 = np.delete(model.params, idx_beta0)
        else:
            raise ValueError('')
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
    
    def add_constant(self, X):
        """Add constant column to the X matrix"""
        if self.need_add_constant == True:
            return sm.add_constant(X, prepend=True)
        else:
            return X

    def detect_constant(self, X_train):
        """statsmodels integrates the intercept in X_train and in beta_hat.
        This function computes the index column of the constant."""
        temp = np.where(np.all(X_train==1., axis=0))[0]
        if temp.shape[0] == 1:
            # There is a constant
            return temp[0]
        elif temp.shape[0] > 1:
            raise ValueError('There is at least 2 contant features in X_train.')
        else:
            return None

    def compute_covariance(self, X_train=None, y_train=None, force=False):
        """Compute the variance-covariance matrix of beta_hat"""
        X_train_origin = X_train
        X_train = self.add_constant(X_train)
        if self.module == 'statsmodels':
            if hasattr(self.model, 'normalized_cov_params') and not force:
                # statsmodels GLM computes the covariance matrix
                self.cov_params = self.model.normalized_cov_params
            else:
                # statsmodels do not support the cov matrix for regularized GLM 
                raise ValueError('Model not supported yet.')
        elif self.module == 'sklearn':
            if False: # Logit
                assert(X_train is not None)
                yhat = clf.predict_proba(X_train_origin)[:,clf.classes_==1]
                W = np.diag((yhat*(1-yhat)).squeeze())
                Xt_W_X = X_train.T.dot(W).dot(X_train)
                unrestricted_cov_params = np.linalg.inv(Xt_W_X)
                self.cov_params = unrestricted_cov_params
            elif self.model.get_params()['penalty']=='l2': #L2 Regularized Logit
                assert(X_train is not None and y_train is not None)
                # we need to fit the unregularized logit to have an estimate of Omega(beta_0)
                # scikit-learn do not yet support unregularized logistic regression.
                # An hacky solution is to set C to a very high value. Note that the 
                # statsmodels implementation is too slow for some ML datasets
                # See: https://github.com/scikit-learn/scikit-learn/issues/6738
                sklearn_params = self.model.get_params()
                sklearn_params['C'] = 1e12
                unregularized_model = linear_model.LogisticRegression()
                unregularized_model.set_params(**sklearn_params)
                unregularized_model.fit(X_train_origin, y_train)
                if unregularized_model.n_iter_ >= sklearn_params['max_iter']:
                    print('Warning! Max number of iterations reached. May not be optimal.')
                yhat_ur = unregularized_model.predict_proba(X_train_origin)[:,unregularized_model.classes_==1]
                W = np.diag((yhat_ur*(1-yhat_ur)).squeeze())
                Xt_W_X = X_train.T.dot(W).dot(X_train)
                alpha = 1.0/self.model.get_params()['C']
                invOmegaAlpha = np.linalg.inv(Xt_W_X + 2*alpha*np.identity(Xt_W_X.shape[0]))
                self.cov_params = invOmegaAlpha.dot(Xt_W_X).dot(invOmegaAlpha)
            else:
                raise ValueError('L1 Regularized Logit not supported yet.')
        else:
            raise Exception('CovarianceNotSupported')

    def compute_orthogonal_projection(self, x, idx_beta0, overshoot = 1e-6):
        """Compute the orthogonal projection of x on the decision hyperplane, which is the 
        optimal L2-adversarial pertubation"""
        beta_hat = self.beta_hat
        beta_hat_var = beta_hat[]
        delta = - (x.dot(beta_hat)/sum(beta_hat_var**2)) * beta_hat_var
        delta = delta * (1 + overshoot)
        delta = np.insert(delta, idx_beta_0, 0) # add back the constant
        self.delta = delta