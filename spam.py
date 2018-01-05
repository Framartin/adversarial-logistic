"""
Spambase Data Set
https://archive.ics.uci.edu/ml/datasets/spambase
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from adversarialLogistic import AdversarialLogistic
from adversarialLogistic import plot_intensity_vs_level
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

ALPHAS = np.arange(0.001, 0.999, 0.001).tolist()
COLORS_MODELS = ['r', 'orange', 'orchid']


#--------------------------------
# I - Data
#--------------------------------

# We create two DataFrame for each objects, because statsmodels.GLM needs the
# constant to be passed as a column.

data = pd.read_csv('data/spam/spambase.data', header=None)
data.rename(columns={57:'spam'}, inplace=True)
y = data.pop('spam')
X = data
X_with_const = sm.add_constant(data)

# stratified CV
X_train, X_test, X_train_with_const, X_test_with_const, y_train, y_test = train_test_split(
    X, X_with_const, y, test_size=0.30, random_state=42, stratify=y)
del X, X_with_const, y # avoid bugs

idx_x0 = 3 #0
x_0 = X_test.iloc[[idx_x0]].as_matrix()
x_0_with_const = X_test_with_const.iloc[[idx_x0]].as_matrix().squeeze()
y_0 = y_test.iloc[[idx_x0]].squeeze()


#------------------------------------------
# II - Train Logistic Regressions
#         and Compute Adversarial Examples
#------------------------------------------

# A - GLM
glm_binom = sm.GLM(y_train, X_train_with_const, family=sm.families.Binomial())
res = glm_binom.fit()
y_pred = (res.predict(exog=X_train_with_const) > 0.5)
glm_acc_is = metrics.accuracy_score(y_train, y_pred)
y_pred = (res.predict(exog=X_test_with_const) > 0.5)
glm_acc_oos = metrics.accuracy_score(y_test, y_pred)

adv_glm = AdversarialLogistic(res, X_train=X_train_with_const,lower_bound=0)
adv_glm.compute_covariance()
x_adv_glm = adv_glm.compute_adversarial_perturbation(x_0_with_const, y_0, alpha=0.95)
print('**GLM**')
print('accuracy in-sample: {0}'.format(glm_acc_is))
print('accuracy out-of-sample: {0}'.format(glm_acc_oos))
print('real y_0: {0}'.format(y_0))
print('predict x_0: {0}'.format(res.predict(exog=x_0_with_const)))
print('predict x_adv_0: {0}'.format(res.predict(exog=x_adv_glm['x_adv_0'])))
print('predict x_adv_star: {0}'.format(res.predict(exog=x_adv_glm['x_adv_star'])))
print('lambda_star: {0}'.format(x_adv_glm['lambda_star']))
#adv_glm.plot_lambda_vs_alpha(x=x_0_with_const, y=y_0, matplotlib=plt, alpha_max = 0.96, label = 'IRLS', color='r')
# explode in alpha=0.98 
# store for future plot
pertubations_glm = adv_glm.compute_adversarial_perturbation(x_0_with_const, y_0, alpha=np.arange(0.04, 0.93, 0.001).tolist(), verbose=False)

# L2 regularized statsmodels
# Commented bc. the computation of cov_params is not implemented.
#glm_L2 = sm.GLM(y, data, family=sm.families.Binomial())
#res_L2 = glm_L2.fit_regularized(alpha=1.0, L1_wt=0.0)
#res_L2.normalized_cov_params

# B - Unregularized sklearn
# scikit-learn do not yet support unregularized logistic regression.
# An hacky solution is to set C to a very high value.
# See: https://github.com/scikit-learn/scikit-learn/issues/6738

lr = linear_model.LogisticRegression(C = 1e12, solver='liblinear', random_state = 1234)
lr.fit(X_train, y_train)
lr_acc_is = lr.score(X = X_train, y = y_train)
lr_acc_oos = lr.score(X = X_test, y = y_test)

# compare estimates between estimation methods
lr.coef_.squeeze()
res.params.as_matrix()

adv_sk = AdversarialLogistic(lr, lower_bound=0)
adv_sk.compute_covariance(X_train = X_train, y_train = y_train)
x_adv_sk = adv_sk.compute_adversarial_perturbation(x_0, y_0, alpha=0.95)
x_adv_sk_0 = x_adv_sk['x_adv_0'][1:].reshape((1, x_adv_sk['x_adv_0'].shape[0]-1))
x_adv_sk_star = x_adv_sk['x_adv_star'][1:].reshape((1, x_adv_sk['x_adv_star'].shape[0]-1))
print('**Unregularized sklearn**')
print('number of iterations: {0}'.format(lr.n_iter_[0]))
print('accuracy in-sample: {0}'.format(lr_acc_is))
print('accuracy out-of-sample: {0}'.format(lr_acc_oos))
print('predict x_0: {0}'.format(lr.predict_proba(x_0)[0,adv_sk.model.classes_==1][0]))
print('predict x_adv_0: {0}'.format(lr.predict_proba(x_adv_sk_0)[0,adv_sk.model.classes_==1][0]))
print('predict x_adv_star: {0}'.format(lr.predict_proba(x_adv_sk_star)[0,adv_sk.model.classes_==1][0]))
print('lambda_star: {0}'.format(x_adv_sk['lambda_star']))
#adv_sk.plot_lambda_vs_alpha(x=x_0, y=y_0, matplotlib=plt, label = 'Unregularized liblinear', color='orange')
# store for future plot
pertubations_sk = adv_sk.compute_adversarial_perturbation(x_0, y_0, alpha=ALPHAS, verbose=False)


# C - L2 regularized sklearn
lr_l2_CV = linear_model.LogisticRegressionCV(penalty = 'l2', solver='liblinear', Cs=100,
    random_state = 42, n_jobs=-1)
# Cs=100 : grid of 100 values
lr_l2_CV.fit(X_train, y_train)
bestC = lr_l2_CV.C_[0]
del lr_l2_CV
# retrain LR with the best C
# this is the same than above, but currently adversarialLogistic 
# doesn't support linear_model.LogisticRegressionCV
lr_l2 = linear_model.LogisticRegression(penalty = 'l2', solver='liblinear', random_state = 42, C=bestC)
lr_l2.fit(X_train, y_train)
lr_l2_acc_is = lr_l2.score(X = X_train, y = y_train)
lr_l2_acc_oos = lr_l2.score(X = X_test, y = y_test)

adv_skl2 = AdversarialLogistic(lr_l2, lower_bound=0)
adv_skl2.compute_covariance(X_train, y_train)
x_adv_skl2 = adv_skl2.compute_adversarial_perturbation(x_0, y_0, alpha=0.95)
x_adv_skl2_0 = x_adv_skl2['x_adv_0'][1:].reshape((1, x_adv_skl2['x_adv_0'].shape[0]-1))
x_adv_skl2_star = x_adv_skl2['x_adv_star'][1:].reshape((1, x_adv_skl2['x_adv_star'].shape[0]-1))
print('**L2-regularized sklearn**')
print('Best C found: {0}'.format(bestC))
print('number of iterations: {0}'.format(lr_l2.n_iter_))
print('accuracy in-sample: {0}'.format(lr_l2_acc_is))
print('accuracy out-of-sample: {0}'.format(lr_l2_acc_oos))
print('predict x_0: {0}'.format(lr_l2.predict_proba(x_0)[0,adv_skl2.model.classes_==1][0]))
print('predict x_adv_0: {0}'.format(lr_l2.predict_proba(x_adv_skl2_0)[0,adv_skl2.model.classes_==1][0]))
print('predict x_adv_star: {0}'.format(lr_l2.predict_proba(x_adv_skl2_star)[0,adv_skl2.model.classes_==1][0]))
print('lambda_star: {0}'.format(x_adv_skl2['lambda_star']))
#adv_skl2.plot_lambda_vs_alpha(x=x_0, y=y_0, matplotlib=plt, label = 'L2-regularized liblinear', color='orchid')
pertubations_skl2 = adv_skl2.compute_adversarial_perturbation(x_0, y_0, alpha=ALPHAS, verbose=False)

# Notes
# some lambda_star_unregularized > lambda_star_l2_regularized, but not for all.


#-----------------------------------
# III - Compare Covariance Matrices 
#-----------------------------------

# Compare the statsmodels IRLS one, and the one of our code using the estimated betas by sklearn 
print('Covariance statsmodels/custom code on sklearn:')

np.all(np.around(res.normalized_cov_params, 4)==np.around(adv_sk.cov_params, 4))
varsAbsDiff = np.max(np.abs(res.normalized_cov_params - adv_sk.cov_params))
np.max(varsAbsDiff) #there is a huge difference between the two
# 1332
np.where(varsAbsDiff > 15)
# there is a huge difference in the estimation of the variance of beta_41
# 4 biggest elements:
temp = np.partition(-varsAbsDiff, 4)
result = -temp[:4]

# there is a big difference between estimates
betasAbsDiff = np.abs(adv_glm.beta_hat - np.around(adv_sk.beta_hat))
np.all(np.around(adv_glm.beta_hat, 4)==np.around(adv_sk.beta_hat, 4))
np.max(betasAbsDiff)
np.mean(betasAbsDiff)

np.column_stack((adv_glm.beta_hat, adv_sk.beta_hat))
np.column_stack((adv_glm.beta_hat[41], adv_sk.beta_hat[41]))

# is the perturbation associated to beta_41 strong?
x_adv_glm['x_adv_0'][41] - x_0_with_const[41]
np.column_stack((x_adv_glm['x_adv_0'], x_0_with_const))
x_adv_glm['x_adv_0']-x_0_with_const

print(res.summary())


#----------------------------------------------------
# IV - Intensity vs Missclassification Level for x_0
#----------------------------------------------------

# Plots of lambda vs alpha for x_0

plot_intensity_vs_level(pertubations_glm, pertubations_sk, pertubations_skl2,
    labels=['IRLS', 'Unregularized liblinear', 'L2-regularized liblinear'], 
    colors=COLORS_MODELS, filename='images/spam_intensities_alphas.png')

plot_intensity_vs_level(pertubations_glm, pertubations_sk, pertubations_skl2,
    labels=['IRLS', 'Unregularized liblinear', 'L2-regularized liblinear'], 
    colors=COLORS_MODELS, ylim=(0.4, 1.7),
    filename='images/spam_intensities_alphas_zoom.png')


#--------------------------------------------------
# V - Densities of the Intensities in the test set
#        accross estimation methods
#--------------------------------------------------

# Density of lambda_star in the test population

ALPHA = 0.90

def compute_lambdas_star(adv, X_test, y_test, alpha, label_model):
    lambdas = []
    for i in range(0, X_test.shape[0]):
        x_0 = X_test.iloc[[i]].as_matrix().squeeze()
        y_0 = y_test.iloc[[i]].squeeze()
        try:
            lambda_star = adv.compute_adversarial_perturbation(x_0, y_0, alpha=alpha, tol_underflow=1e-9,verbose_bounds=False)['lambda_star']
        except ArithmeticError:
            print('Underflow')
            continue
        lambdas.append(lambda_star)
    df_lambdas = pd.DataFrame({
        'lambdas': lambdas,
        'model': [label_model]*len(lambdas)
    })
    return df_lambdas

lambdas_glm = compute_lambdas_star(adv = adv_glm, X_test = X_test_with_const, y_test = y_test, alpha = ALPHA, label_model = 'IRLS')
lambdas_sk = compute_lambdas_star(adv = adv_sk, X_test = X_test, y_test = y_test, alpha = ALPHA, label_model = 'Unregularized liblinear')
lambdas_skl2 = compute_lambdas_star(adv = adv_skl2, X_test = X_test, y_test = y_test, alpha = ALPHA, label_model = 'L2-regularized liblinear')
 
# violinplot
df_lambdas = pd.concat([lambdas_glm, lambdas_sk, lambdas_skl2])
plt.close()
fig = plt.figure(figsize=(7, 5), dpi=150)
sns.violinplot(x=df_lambdas["model"], y=df_lambdas["lambdas"], palette=COLORS_MODELS, gridsize=1000, scale_hue=False, saturation=0.9)
plt.xlabel('Estimation Method')
plt.ylabel('Intensity of the pertubation (λ)')
plt.savefig('images/spam_violinplot.png')
plt.savefig('images/spam_violinplot.pdf')
plt.ylim((-2,9))
plt.savefig('images/spam_violinplot_zoom.png')
plt.savefig('images/spam_violinplot_zoom.pdf')


#---------------------------------------------------
# VI - Quantiles of the Intensities in the test set
#       vs L2-Regularization
#---------------------------------------------------

# Distributions of lambda vs l2-regularization hyperparameter

# plot accuracy, and quantiles of lambdas computed for alpha = 0.90, versus lamdba_l2 (regularization hyperparameter)
num_points = 300
C_range = np.geomspace(start = 1e-7, stop = 1e7, num = num_points)
accuracies_C = pd.DataFrame()
lambdas_C = pd.DataFrame()

print('Effect of Regularization HP...')
for i, C_ in enumerate(C_range):
    print('[{0}] Value C: {1}'.format(i, C_))
    lr_l2_C = linear_model.LogisticRegression(penalty = 'l2', solver='liblinear', C=C_)
    lr_l2_C.fit(X_train, y_train)
    acc_is = lr_l2_C.score(X = X_train, y = y_train)
    acc_oos = lr_l2_C.score(X = X_test, y = y_test)
    lambda_l2 = 1.0/C_
    accuracies_C = accuracies_C.append(pd.DataFrame({'C_': C_, 'lambda_l2': lambda_l2, 
        'acc_is': acc_is, 'acc_oos': acc_oos}, index=[0]), ignore_index = True)
    adv_skl2_C = AdversarialLogistic(lr_l2_C, lower_bound=0)
    adv_skl2_C.compute_covariance(X_train, y_train)
    lambdas_skl2_C = compute_lambdas_star(adv = adv_skl2_C, X_test = X_test, 
        y_test = y_test, alpha = ALPHA, label_model = lambda_l2)
    lambdas_C = lambdas_C.append(lambdas_skl2_C, ignore_index = True)

# draw the evolution of the median, 1rt decile and last decile of lambdas in the test set 
# over values of L2 regularization hyperparameter
sns.reset_orig()
plt.figure(figsize=(7, 5), dpi=150)
plt.xlim(lambdas_C['model'].min(), lambdas_C['model'].max())
plt.xscale('log')
plt.plot(accuracies_C['lambda_l2'], accuracies_C['acc_oos'], linewidth=2, linestyle='--')
medians = lambdas_C.groupby('model').median()
plt.plot(medians.index, medians.lambdas, linewidth=2, color='#B22400')
firstdecile = lambdas_C.groupby('model').quantile(q=0.1)
lastdecile = lambdas_C.groupby('model').quantile(q=0.9)
plt.fill_between(firstdecile.index, firstdecile.lambdas, lastdecile.lambdas, alpha=0.25, linewidth=0, color='#B22400')
plt.xlabel('L2-Regularization Hyperparameter')
#plt.ylabel('Intensity of the pertubation (λ)')
legend = plt.legend(["Out-of-Sample Accuracy", "Intensity of the pertubation (λ)"], loc=3);
legend.get_frame()
plt.savefig('images/spam_intensities_regularization.png')
plt.savefig('images/spam_intensities_regularization.pdf')
