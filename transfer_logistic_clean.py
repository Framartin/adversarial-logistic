"""
CLEAN VERSION

TODO:
- test Binomial negative to account for overdispersion??
- test scale parameter for overdispersion?

"""

import statsmodels.api as sm
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import special, stats

data = sm.datasets.star98.load()
data.exog = sm.add_constant(data.exog, prepend=False)

glm_binom = sm.GLM(data.endog, data.exog, family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())

# example to perturbate
#x_0 = data.exog[4,:]
#x_0 = data.exog[6,:]
x_0 = data.exog[1,:]
print('pred sane: {0}'.format(res.predict(x_0)))
# >>> res.predict(data.exog[4,:])
# array([ 0.32251021])
# >>> res.predict(data.exog[4,:], linear=True)
beta_hat = res.params
idx_beta_0 = 20 # column number corresponding to beta_0
beta_hat_var = beta_hat[np.arange(len(beta_hat))!=idx_beta_0] # beta_hat without the constant (noted w in ML)
var_covar_matrix = res.normalized_cov_params

#import pdb; pdb.set_trace()

def compute_adv_pertubation(x, beta_hat, beta_hat_var, idx_beta_0, overshoot = 0.0):
    eps = - (x.dot(beta_hat)/sum(beta_hat_var**2)) * beta_hat_var
    eps = eps * (1 + overshoot)
    eps = np.insert(eps, idx_beta_0, 0) # add back the constant
    return eps 

alpha = 0.05
eps = compute_adv_pertubation(x_0, beta_hat, beta_hat_var, idx_beta_0)

preds = [(x + compute_adv_pertubation(x, beta_hat, beta_hat_var, idx_beta_0)).dot(beta_hat) for x in data.exog]

def solve_lambda(alpha, x, beta_hat, eps, tolerance = 1e-8, verbose = False):
    if verbose:
        print('-----------')
    d = math.sqrt(2)*special.erfinv(2*alpha-1)
    A = np.outer(beta_hat, beta_hat) - d**2 * var_covar_matrix 
    a = eps.dot(A).dot(eps)
    b = x.dot(A).dot(eps) + eps.dot(A).dot(x)
    c = x.dot(A).dot(x)
    #import pdb; pdb.set_trace()
    if verbose:
        print('value a: {0}'.format(a))
    delta = b**2 - 4*a*c
    if delta < 0:
        if verbose:
            print('No real solution. Delta: {0}'.format(delta))
        return None
    elif delta == 0:
        if verbose:
            print('One solution')
        return -b/(2*a)
    elif delta > 0:
        lambda1 = (-b - delta**0.5) / (2*a)
        lambda2 = (-b + delta**0.5) / (2*a)
        if verbose:
            print('Two solutions: {0}, {1}'.format(lambda2, lambda1))
        for lambda_star in [lambda1, lambda2]: #[lambda1, lambda2]: # TODO: verifier que resiste a l'ordre
            x_adv = x+lambda_star*eps
            eq = abs(x_adv.dot(beta_hat) + d*math.sqrt( x_adv.dot(var_covar_matrix).dot(x_adv)))
            # TODO: on est loin
            if verbose:
                print('Value eq: {0}'.format(eq))
            eq2 = abs(x_adv.dot(A).dot(x_adv))
            #print('Value eq2: {0}'.format(eq2))
            #print('--')
            if eq < tolerance:
                if verbose:
                    print('----')
                return lambda_star
        import pdb; pdb.set_trace()
        raise ValueError('Error when solving the 2nd degres eq')

#solve_lambda(alpha, x_0, beta_hat, eps, verbose = True)
#solve_lambda(0.3, x_0, beta_hat, eps, verbose = True)
#solve_lambda(0.7, x_0, beta_hat, eps, verbose = True)


def plot_alpha_lambda(x, min_alpha=0, max_alpha=1, step=0.01, tolerance=1e-8, verbose=False):
    eps = compute_adv_pertubation(x, beta_hat, beta_hat_var, idx_beta_0)
    alpha_range = np.arange(min_alpha, max_alpha, step)
    y_lambda = [solve_lambda(alpha, x, beta_hat, eps, tolerance=tolerance, verbose=verbose) for alpha in alpha_range] # , math.inf
    plt.plot(alpha_range, y_lambda, 'r--')
    plt.show()

#plot_alpha_lambda(x_0)
#plot_alpha_lambda(x_0, min_alpha = 0.001, max_alpha=0.5, step = 0.001)
#plot_alpha_lambda(x_0, min_alpha = 0.501, max_alpha=0.999, step = 0.001)

#plot_alpha_lambda(data.exog[2,:], min_alpha = 0.501, max_alpha=0.999, step = 0.001)

#plot_alpha_lambda(data.exog[4,:], min_alpha = 0.501, max_alpha=0.999, step = 0.001)

plot_alpha_lambda(data.exog[1,:], verbose=True)
plot_alpha_lambda(data.exog[4,:], verbose=True)

class AdversarialGLM():
    """AdversarialGLM"""
    def __init__(self, arg):
        super(AdversarialGLM, self).__init__()
        self.arg = arg
    
    def test(self):
        print(self.arg)


#def f(alpha, x_0 = x_0, beta_hat_0 = beta_hat_0, var_covar_matrix = var_covar_matrix, c = c):
#    x_1 = x_0 + alpha * c
#    numerator = np.dot(x_1, beta_hat_0)
#    return  numerator / math.sqrt(np.dot(np.dot(np.transpose(x_1), var_covar_matrix), x_1))
# 
# def plot_f(min_a = 0.0, max_a = 10.0, step = 0.001, hline=None):
#     alpha_range = np.arange(min_a, max_a, step)
#     y_alpha = [f(alpha) for alpha in alpha_range]
#     plt.plot(alpha_range, y_alpha, 'r--')
#     if hline is not None:
#         plt.axhline(y = d)
#     plt.show()

