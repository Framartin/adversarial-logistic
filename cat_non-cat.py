### DOGS vs CATS
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy import ndimage
#from lr_utils import load_dataset
import statsmodels.api as sm
from sklearn import linear_model
from adversarialLogistic import AdversarialLogistic
import pickle
import glob
#from PIL import Image
from scipy import misc
import os

TRAIN_DIR = 'data/cats/data64/train'

# https://www.kaggle.com/c/dogs-vs-cats/data

def import_images():
    train_cats = sorted(glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg')))
    train_dogs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'dog*.jpg')))
    train_cats = train_cats[0:100] #TODO
    train_dogs = train_dogs[0:100] #TODO
    n = len(train_dogs) + len(train_dogs)
    y = np.zeros((1,n))
    images = []
    for image_path in train_cats:
        images.append(misc.imread(image_path, mode='RGB'))
    y[0:len(train_cats)] = 1
    for image_path in train_dogs:
        images.append(misc.imread(image_path, mode='RGB'))
    X = np.asarray(images)
    print(images[1].dtype)
    print('Imported: {0} cats and {1} dogs.'.format(len(train_cats), len(train_dogs)))
    return X, y

X, y = import_images()

#train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
#index = 25
#plt.imshow(train_set_x_orig[index])
#plt.show()
X_train = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).squeeze()
X_test = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).squeeze()
X_train = X_train/255
X_test = X_test/255
#X_train = sm.add_constant(X_train, prepend=True)
#X_test = sm.add_constant(X_test, prepend=True)
y_train = train_set_y.T
y_test = test_set_y.T

del train_set_x_orig, train_set_y, test_set_x_orig, test_set_y

#glm_binom = sm.GLM(y, data, family=sm.families.Binomial())
#res = glm_binom.fit()
#res = glm_binom.fit_regularized()

lr_l2_CV = linear_model.LogisticRegressionCV(penalty = 'l2', random_state = 42)
lr_l2_CV.fit(X_train, y_train)
bestC = lr_l2_CV.C_
print('C: {0}'.format(bestC))

lr_l2 = linear_model.LogisticRegression(penalty = 'l2', random_state = 42, C=bestC)
lr_l2.fit(X_train, y_train)

adv = AdversarialLogistic(lr_l2, lower_bound=0, upper_bound=255)
print(adv.beta_hat.shape)
#adv.compute_covariance(X_train, y_train)

#with open('adv.pkl', 'wb') as output:
#    pickle.dump(adv, output, pickle.HIGHEST_PROTOCOL)

with open('adv.cov_params.pkl', 'rb') as input:
    adv.cov_params = pickle.load(input)


x_0 = X_test[0,:].squeeze()
y_0 = y_test[0,:].squeeze()
#adv.plot_lambda_vs_alpha(x_0)

x_adv = adv.compute_adversarial_perturbation(x_0, y_0, alpha=0.95, out_bounds='clipping')

#fig = plt.figure()
#plt.figure()
#a = fig.add_subplot(2,2,1)
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(x_0.reshape(64,64,3))
axarr[0,0].set_title('Benign example')
axarr[0,1].imshow(x_adv['x_adv_0'][1:].reshape(64,64,3))
axarr[0,1].set_title('Orthogonal example')
axarr[1,0].imshow(x_adv['x_adv_star'][1:].reshape(64,64,3))
axarr[1,0].set_title('0.95-logistic adversarial example')
delta_star_plot = np.abs(x_adv['x_adv_star'][1:].reshape(64,64,3) - x_0.reshape(64,64,3))
axarr[1,1].imshow(delta_star_plot)
axarr[1,1].set_title('0.95-logistic adversarial perturbation')
plt.show()