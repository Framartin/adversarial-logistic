"""

TODO:
- use one subclass for each model type for cleaner implementation
- fix bug: handle the case where, for sklearn, if the constant is already on X_train, beta0 is inside model.coef_
- remove dependance on statsmodels
- add documentation
- improve plot alpha
- fix computation of alpha
"""

import statsmodels.api as sm
from sklearn import linear_model
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import special, stats

#import pdb; pdb.set_trace()

class AdversarialLogistic(object):
    """docstring for AdversarialLogistic"""

    def __init__(self, model, X_train=None, lower_bound=float('-inf'), upper_bound=float('inf')):
        super(AdversarialLogistic, self).__init__()
        #TODO:
        # check if model is supported
        # check that not multinomial logit
        # check that we use the bernoouilli Logit, not the Binomial
        self.model = model
        module = getattr(model, '__module__')
        self.X_has_constant = False # X_train and x includes a constant terms
        self.model_has_intercept = False # model was trained with an intercept
        if module == 'sklearn.linear_model.logistic':
            self.module = 'sklearn'
            if model.get_params()['fit_intercept']:
                self.model_has_intercept = True
                # model has intercept, but constant not in X_train nor in x 
                self.beta_hat_minus0 = model.coef_.squeeze() #[:,1:].squeeze()
                self.beta_hat = np.insert(model.coef_, 0, model.intercept_)
                #self.beta_hat = model.coef_.squeeze()
                self.idx_beta0 = 0
            else:
                self.beta_hat_minus0 = self.beta_hat = model.coef_.squeeze()
        elif module == 'statsmodels.genmod.generalized_linear_model':
            self.module = 'statsmodels'
            assert(X_train is not None)
            self.beta_hat = model.params.as_matrix().squeeze()
            #idx_beta0 = self.detect_constant(X_train)
            if 'const' not in model.params.index:
                # No constant in (X, beta_hat) 
                self.beta_hat_minus0 = model.params
            else:
                # model has intercept, and a constant is in X_train and in x 
                self.model_has_intercept = True
                self.X_has_constant = True
                self.beta_hat_minus0 = model.params.drop('const').as_matrix().squeeze()
                self.idx_beta0 = np.where('const' == model.params.index)[0].squeeze()
        else:
            raise ValueError('model not supported.')
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
    
    def add_constant(self, X=None, x=None):
        """Add constant column to the X matrix if needed"""
        if self.model_has_intercept and not self.X_has_constant:
            if X is not None:
                return sm.add_constant(X, prepend=True)
            elif x is not None:
                return np.insert(x, 0, 1)
        else:
            if X is not None:
                return X
            elif x is not None:
                return x

    def detect_constant(self, X_train):
        """statsmodels integrates the intercept in X_train and in beta_hat.
        This function computes the index column of the constant."""
        temp = np.where(np.all(X_train==1., axis=0))[0]
        if temp.shape[0] == 1:
            # There is a constant
            return temp[0]
        elif temp.shape[0] > 1:
            raise ValueError('There is at least 2 constant features in X_train.')
        else:
            return None

    def compute_covariance(self, X_train=None, y_train=None, force=False):
        """Compute the variance-covariance matrix of beta_hat"""
        #TODO: code force param properly
        X_train_origin = X_train
        X_train = self.add_constant(X_train)
        if self.module == 'statsmodels':
            if hasattr(self.model, 'normalized_cov_params') and not force:
                # statsmodels GLM computes the covariance matrix
                self.cov_params = self.model.normalized_cov_params
            else:
                # statsmodels do not support the computation of the cov matrix for regularized GLM 
                raise ValueError('Model not supported yet.')
        elif self.module == 'sklearn':
            if self.model.get_params()['C']>=1e10: # Logit without regularization
                assert(X_train is not None)
                yhat = self.model.predict_proba(X_train_origin)[:,self.model.classes_==1]
                del X_train_origin
                W = np.diag((yhat*(1-yhat)).squeeze())
                Xt_W_X = X_train.T.dot(W).dot(X_train)
                del X_train
                self.cov_params = np.linalg.inv(Xt_W_X) # unrestricted Var(beta)
            elif self.model.get_params()['penalty']=='l2': #L2 Regularized Logit
                assert(X_train is not None and y_train is not None)
                # we need to fit the unregularized logit to have an estimate of Omega(beta_0)
                # scikit-learn do not yet support unregularized logistic regression.
                # An hacky solution is to set C to a very high value. Note that the 
                # statsmodels implementation is too slow for some ML datasets
                # See: https://github.com/scikit-learn/scikit-learn/issues/6738
                # Question: is there a better way to compute Var(beta) without the unrestricted estimation?
                sklearn_params = self.model.get_params()
                sklearn_params['C'] = 1e12
                unregularized_model = linear_model.LogisticRegression()
                unregularized_model.set_params(**sklearn_params)
                unregularized_model.fit(X_train_origin, y_train)
                if unregularized_model.n_iter_ >= sklearn_params['max_iter']:
                    print('Warning! Max number of iterations reached. May not be optimal.')
                yhat_ur = unregularized_model.predict_proba(X_train_origin)[:,unregularized_model.classes_==1]
                del X_train_origin
                W = np.diag((yhat_ur*(1-yhat_ur)).squeeze())
                Xt_W_X = X_train.T.dot(W).dot(X_train)
                del X_train
                lambda_c = 1.0/self.model.get_params()['C']
                invOmegaLambda = np.linalg.inv(Xt_W_X + 2*lambda_c*np.identity(Xt_W_X.shape[0]))
                self.cov_params = invOmegaLambda.dot(Xt_W_X).dot(invOmegaLambda)
            else:
                raise ValueError('L1 Regularized Logit not supported yet.')
        else:
            raise Exception('CovarianceNotSupported')

    def compute_orthogonal_projection(self, x, overshoot = 1e-6):
        """Compute the orthogonal projection of x on the decision hyperplane, which is the 
        optimal L2-adversarial pertubation.

        Parameters
        ----------
        x : array_like
            1-D array of the example to perturbate.
        overshoot : float
            Multiplies the adversarial pertubation by (1 + overshoot) to overcome underflow issues. 
        """
        beta_hat = self.beta_hat
        beta_hat_minus0 = self.beta_hat_minus0
        delta = - (x.dot(beta_hat)/sum(beta_hat_minus0**2)) * beta_hat_minus0
        delta = delta * (1 + overshoot)
        if self.model_has_intercept:
            delta = np.insert(delta, self.idx_beta0, 0) # add back the constant
        return delta

    def compute_alpha(self, alpha, y):
        if np.sign(y) == 1:
            return alpha
        else:
            return 1-alpha

    def solve_lambda(self, alpha, x, delta, tol = 1e-6, verbose = False):
        if verbose:
            print('-----------')
        beta_hat = self.beta_hat
        d = math.sqrt(2)*special.erfinv(2*alpha-1)
        A = np.outer(beta_hat, beta_hat) - d**2 * self.cov_params 
        a = delta.dot(A).dot(delta)
        b = x.dot(A).dot(delta) + delta.dot(A).dot(x)
        c = x.dot(A).dot(x)
        if verbose:
            print('value a: {0}'.format(a))
        DeltaEq2 = b**2 - 4*a*c
        if abs(DeltaEq2) < tol: # DeltaEq2 == 0 
            if verbose:
                print('One solution')
            return -b/(2*a)
        elif DeltaEq2 < 0:
            if verbose:
                print('No real solution. Delta: {0}'.format(DeltaEq2))
            return None
        elif DeltaEq2 > 0:
            lambda1 = (-b - DeltaEq2**0.5) / (2*a)
            lambda2 = (-b + DeltaEq2**0.5) / (2*a)
            if verbose:
                print('Two solutions: {0}, {1}'.format(lambda2, lambda1))
            for lambda_star in [lambda1, lambda2]: #[lambda1, lambda2]: # TODO: verifier que resiste a l'ordre
                x_adv = x + lambda_star*delta
                eq = abs(x_adv.dot(beta_hat) + d*math.sqrt( x_adv.dot(self.cov_params).dot(x_adv)))
                if verbose:
                    print('Value eq: {0}'.format(eq))
                if eq < tol:
                    if verbose:
                        print('----')
                    return lambda_star
        raise ValueError('Error when solving the 2nd degree equation.')

    def plot_lambda_vs_alpha(self, x, y, alpha_min=0.001, alpha_max=0.999, step=0.01, tol=1e-6, verbose=False):
        if not (hasattr(self, 'cov_params')):
            raise Exception('Missing cov_params. Call: self.compute_covariance(X_train, y_train)')
        x = self.add_constant(x=x)
        delta = self.compute_orthogonal_projection(x)
        alpha_range = np.arange(alpha_min, alpha_max, step)
        alpha_range = [self.compute_alpha(alpha, y) for alpha in alpha_range]
        lambdas = [self.solve_lambda(alpha, x, delta, tol=tol, verbose=verbose) for alpha in alpha_range]
        plt.style.use('ggplot') #bmh
        plt.plot(alpha_range, lambdas)
        plt.xlabel('Missclassification level (α)')
        plt.ylabel('Intensity of the pertubation (δ)')
        plt.show()

    def check_bounds(self, x_adv, out_bounds):
        assert(out_bounds in ['clipping', 'missing', 'nothing'])
        if np.any(x_adv < self.lower_bound):
            print('Adversarial example x_adv < lower_bound.')
            if out_bounds == 'missing':
                return None
            elif out_bounds == 'clipping':
                x_adv[x_adv < self.lower_bound] = self.lower_bound
        if np.any(x_adv > self.upper_bound):
            print('Adversarial example x_adv > upper_bound.')
            if out_bounds == 'missing':
                return None
            elif out_bounds == 'clipping':
                x_adv[x_adv > self.upper_bound] = self.upper_bound
        return x_adv

    def compute_adversarial_perturbation(self, x, y, alpha = 0.05, out_bounds='nothing', tol=1e-6, verbose=False):
        # param out_bounds: 'clipping' or 'missing' or 'nothing'
        #TODO: add tol to check range
        x = self.add_constant(x=x)
        if not (hasattr(self, 'cov_params')):
            raise Exception('Missing cov_params. Call: self.compute_covariance(X_train, y_train)')
        alpha = self.compute_alpha(alpha, y)
        delta = self.compute_orthogonal_projection(x)
        x_adv_0 = x + delta
        # check range of x_adv_0
        x_adv_0 = self.check_bounds(x_adv_0, out_bounds)
        # check pred(x_adv_0)
        assert(np.sign(x_adv_0.dot(self.beta_hat)) != np.sign(y))
        lambda_star = self.solve_lambda(alpha, x, delta, tol=tol, verbose=verbose)
        x_adv_star = x + lambda_star * delta
        # check range of x_adv_star
        x_adv_star = self.check_bounds(x_adv_star, out_bounds)
        # check pred(x_adv_star)
        if alpha > 0.5:
            assert(np.sign(x_adv_star.dot(self.beta_hat)) != np.sign(y))
        return {'lambda_star': lambda_star, 'x_adv_star': x_adv_star, 'x_adv_0': x_adv_0}
