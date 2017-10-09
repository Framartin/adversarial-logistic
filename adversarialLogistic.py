import statsmodels.api as sm
from sklearn import linear_model
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import special, stats


class AdversarialLogit(object):
    """docstring for AdversarialLogit"""

    def __init__(self, model, lower_bound=float('-inf'), upper_bound=float('inf')):
        super(AdversarialLogit, self).__init__()
        # check if model is supported
        # check that not multinomial logit
        # check that we use the bernoouilli Logit, not the Binomial
        self.model = model
        module = getattr(model, '__module__')
        if(module == sklearn.linear_model.logistic):
            self.module = 'sklearn'
            self.beta = model.coef_
            self.intercept_ # TODO
        elif(self.module == 'statsmodels'):
            self.module = 'statsmodels'
            self.beta = model.params
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.penalty=False # 'l1' or 'l2'

    
    def compute_covariance(self, X_train=None, y_train=None, force=False):
        if self.module == 'statsmodels':
            if hasattr(self.model, 'normalized_cov_params') and not force:
                # statsmodels GLM computes the covariance matrix
                self.cov_params = self.model.normalized_cov_params
            else:
                # statsmodels do not support the cov matrix for regularized GLM 
                raise ValueError('Model not supported yet.')
        elif self.module == 'sklearn':
            if True: # Logit
                yhat = clf.predict_proba(X_train)[:,clf.classes_==1]
                W = np.diag((yhat*(1-yhat)).squeeze())
                Xt_W_X = X_train.T.dot(W).dot(X_train)
                unrestricted_cov_params = np.linalg.inv(Xt_W_X)
                self.cov_params = unrestricted_cov_params
            elif self.model.get_params('penalty')=='l2': #L2 Regularized Logit
                # we need to fit the unregularized logit to have an estimate of Omega(beta_0)
                # scikit-learn do not yet support unregularized logistic regression.
                # An hacky solution is to set C to a very high value. Note that the 
                # statsmodels implementation is too slow for some ML datasets
                # See: https://github.com/scikit-learn/scikit-learn/issues/6738
                sklearn_params = self.model.get_params()
                sklearn_params['C'] = 1e12
                unregularized_model = linear_model.LogisticRegression()
                unregularized_model.set_params(**sklearn_params)
                unregularized_model.fit(X_train, y_train)
                if unregularized_model.n_iter_ >= sklearn_param['max_iter']:
                    print('Warning! Max number of iterations reached. May not be optimal.')
                yhat_ur = unregularized_model.predict_proba(X_train)[:,clf.classes_==1]
                W = np.diag((yhat_ur*(1-yhat_ur)).squeeze())
                Xt_W_X = X_train.T.dot(W).dot(X_train)
                alpha = 1.0/self.model.get_params()['C']
                invOmegaAlpha = np.linalg.inv(Xt_W_X + 2*alpha*np.identity(Xt_W_X.shape[0]))
                self.cov_params = invOmegaAlpha.dot(Xt_W_X).dot(invOmegaAlpha)
            else:
                raise ValueError('L1 Regularized Logit not supported yet.')
        else:
            raise Exception('CovarianceNotSupported')

    def compute_orthogonal_projection(x, beta_hat, beta_hat_var, idx_beta_0, overshoot = 1e-6):
        eps = - (x.dot(beta_hat)/sum(beta_hat_var**2)) * beta_hat_var
        eps = eps * (1 + overshoot)
        eps = np.insert(eps, idx_beta_0, 0) # add back the constant
        self.eps = eps