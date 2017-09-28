"""
POC of a idea to improve transferability of 
adversarial examples for logistic regression

TODO:
- extend to non-binary classification
- non-continuous variables
- write in this style: x_0.dot(beta_hat_0)

http://www.statsmodels.org/dev/examples/notebooks/generated/glm.html
http://www.statsmodels.org/dev/glm.html
"""

import statsmodels.api as sm
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import special

data = sm.datasets.star98.load()
data.exog = sm.add_constant(data.exog, prepend=False)

glm_binom = sm.GLM(data.endog, data.exog, family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())

# example to perturbate
x_0 = data.exog[4,:]
# >>> res.predict(data.exog[4,:])
# array([ 0.32251021])
beta_hat_0 = res.params
var_covar_matrix = res.normalized_cov_params # TODO: this is the normalized covar matrix.


def compute_adv_pertubation(x, beta):
    return np.dot((np.dot(x, beta))/(sum(beta * beta)), beta)

x_adv_0 = compute_adv_pertubation(x_0, beta_hat_0)

alpha = 0.05 # TODO: unify notation
c = compute_adv_pertubation(x_0, beta_hat_0)

def f(alpha, x_0 = x_0, beta_hat_0 = beta_hat_0, var_covar_matrix = var_covar_matrix, c = c):
    x_1 = x_0 + alpha * c
    numerator = np.dot(x_1, beta_hat_0)
    return  numerator / math.sqrt(np.dot(np.dot(np.transpose(x_1), var_covar_matrix), x_1))

def plot_f(min_a = 0.0, max_a = 10.0, step = 0.001, hline=None):
    alpha_range = np.arange(min_a, max_a, step)
    y_alpha = [f(alpha) for alpha in alpha_range]
    plt.plot(alpha_range, y_alpha, 'r--')
    if hline is not None:
        plt.axhline(y = d)
    plt.show()

d = -math.sqrt(2)*special.erfinv(2*alpha-1)

plot_f(hline = d)
#plot_f(0.0, 0.05, 0.0000001)

def solve_lambda(beta_hat_0, x_0, c, d, var_covar_matrix, verbose=True):
    A = c.dot(beta_hat_0)**2 - d**2 * ( c.dot(var_covar_matrix).dot(c) )
    #A = np.dot(c, beta_hat_0) ** 2 - d ** 2 * np.dot(np.dot(c, var_covar_matrix), c)
    B = 2 * np.dot(x_0, beta_hat_0) * np.dot(c, beta_hat_0) - d ** 2 * ( np.dot(np.dot(x_0, var_covar_matrix), c) + np.dot(np.dot(c, var_covar_matrix), x_0) )
    C = np.dot(x_0,  beta_hat_0) ** 2 - d ** 2 * np.dot(np.dot(x_0, var_covar_matrix), x_0)
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
        
        #if min(lambda1, lambda2) < 1: #TODO
        #    return max(lambda1, lambda2)
        #else:
        #    return min(lambda1, lambda2)
        return(max(lambda1, lambda2))

lambda_star = solve_lambda(beta_hat_0, x_0, c, d, var_covar_matrix)
print(lambda_star)

## values of lambda VS alpha 

def compute_lambda(alpha, beta_hat_0, x_0, c, var_covar_matrix, verbose=False):
    d = -2*special.erfinv(2*alpha-1)
    lambda_star = solve_lambda(beta_hat_0, x_0, c, d, var_covar_matrix, verbose=verbose)
    return lambda_star

def plot_alpha_lambda(min_a = 0.001, max_a=1, step = 0.01):
    alpha_range = np.arange(min_a, max_a, step)
    y_lambda = [compute_lambda(alpha, beta_hat_0, x_0, c, var_covar_matrix) for alpha in alpha_range]
    plt.plot(alpha_range, y_lambda, 'r--')
    plt.show()

#plot_alpha_lambda()
plot_alpha_lambda(min_a = 0.001, max_a=0.5, step = 0.001)


compute_lambda(0.004, beta_hat_0, x_0, c, var_covar_matrix, verbose = True)