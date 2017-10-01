"""
CLEAN VERSION
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
x_0 = data.exog[4,:]
# >>> res.predict(data.exog[4,:])
# array([ 0.32251021])
# >>> res.predict(data.exog[4,:], linear=True)
beta_hat = res.params
idx_beta_0 = 20 # column number corresponding to beta_0
beta_hat_var = beta_hat[np.arange(len(beta_hat))!=idx_beta_0] # beta_hat without the constant (noted w in ML)
#var_covar_matrix = res.normalized_cov_params # TODO: this is the normalized covar matrix.

W = np.diag(res.fittedvalues*(1-res.fittedvalues))
xt_w_x = data.exog.T.dot(W).dot(data.exog)
var_covar_matrix = np.linalg.inv(xt_w_x)

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
    d = math.sqrt(2)*special.erfinv(2*alpha-1)
    A = np.outer(beta_hat, beta_hat) - d**2 * var_covar_matrix 
    a = eps.dot(A).dot(eps)
    b = x.dot(A).dot(eps) + eps.dot(A).dot(x)
    c = x.dot(A).dot(x)
    if verbose:
        print('value a: {0}'.format(a))
    delta = b**2 - 4*a*c
    if delta < 0:
        if verbose:
            print('No real solution')
        return None
    elif delta == 0:
        if verbose:
            print('One solution')
        return -b/(2*a)
    elif delta > 0:
        lambda1 = (-b - delta**0.5) / (2*a)
        lambda2 = (-b + delta**0.5) / (2*a)
        if verbose:
            print('Two solutions: {0}, {1}'.format(lambda1, lambda2))
        for lambda_star in [lambda1, lambda2]:
            x_adv = x+lambda_star*eps
            eq = abs(x_adv.dot(beta_hat) + d*math.sqrt( x_adv.dot(var_covar_matrix).dot(x_adv)))
            print(eq)
            if eq < tolerance:
                return lambda_star
        raise ValueError('Error when solving the 2nd degres eq')

solve_lambda(alpha, x_0, beta_hat, eps, verbose = True)
solve_lambda(0.3, x_0, beta_hat, eps, verbose = True)


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



d = -math.sqrt(2)*special.erfinv(2*alpha-1)

#plot_f(hline = d)
#plot_f(0.0, 0.05, 0.0000001)

def solve_lambda(beta_hat, x_0, eps, d, var_covar_matrix, verbose=True):
    A = (eps.dot(beta_hat))**2 - d**2 * ( eps.dot(var_covar_matrix).dot(eps) )
    #A = np.dot(eps, beta_hat) ** 2 - d ** 2 * np.dot(np.dot(eps, var_covar_matrix), eps)
    B = 2 * np.dot(x_0, beta_hat) * np.dot(eps, beta_hat) - d ** 2 * ( np.dot(np.dot(x_0, var_covar_matrix), eps) + np.dot(np.dot(eps, var_covar_matrix), x_0) )
    C = np.dot(x_0,  beta_hat) ** 2 - d ** 2 * np.dot(np.dot(x_0, var_covar_matrix), x_0)
    if verbose:
        print('value A: {0}'.format(A))
    delta = B ** 2 - 4 * A * C
    if delta < 0:
        if verbose:
            print('No solution')
        return None
    elif delta == 0:
        if verbose:
            print('One solution')
        return -B/(2*A)
    elif delta > 0:
        lambda1 = (-B - delta**0.5) / (2*A)
        lambda2 = (-B + delta**0.5) / (2*A)
        if verbose:
            print('Two solutions: {0}, {1}'.format(lambda1, lambda2))

        return(max(lambda1, lambda2))

lambda_star = solve_lambda(beta_hat, x_0, eps, d, var_covar_matrix)
print(lambda_star)

## values of lambda VS alpha 

def compute_lambda(alpha, beta_hat_0, x_0, eps, var_covar_matrix, verbose=False):
    d = -2*special.erfinv(2*alpha-1)
    lambda_star = solve_lambda(beta_hat, x_0, eps, d, var_covar_matrix, verbose=verbose)
    return lambda_star

def plot_alpha_lambda(min_a = 0.001, max_a=1, step = 0.01):
    alpha_range = np.arange(min_a, max_a, step)
    y_lambda = [compute_lambda(alpha, beta_hat, x_0, eps, var_covar_matrix) for alpha in alpha_range]
    plt.plot(alpha_range, y_lambda, 'r--')
    plt.show()

#plot_alpha_lambda()
plot_alpha_lambda(min_a = 0.001, max_a=0.5, step = 0.001)


compute_lambda(0.004, beta_hat, x_0, eps, var_covar_matrix, verbose = True)
