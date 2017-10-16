"""
DOGS vs CATS

https://www.kaggle.com/c/dogs-vs-cats/data
"""

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
TEST_DIR = 'data/cats/data64/test'
# cats and dogs that don't like identity:
# squared images without gray bands
IDS_TEST_EXAMPLES = [('dog', 2), ('cat', 5), ('dog', 21), ('cat', 28), ('cat', 45), ('cat', 58), ('cat', 90)]
# values of alphas to compute adversarial examples
ALPHAS = [0.75, 0.9, 0.95]

# TODO: account for the impossibility to perturbate gray bars

def import_train_images():
    train_cats = sorted(glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg')))
    train_dogs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'dog*.jpg')))
    train_cats = train_cats[0:100] #TODO
    train_dogs = train_dogs[0:100] #TODO
    n = len(train_cats) + len(train_dogs)
    y = np.zeros((1,n))
    images = []
    for image_path in train_cats:
        images.append(misc.imread(image_path, mode='RGB'))
    y[0, 0:len(train_cats)] = 1
    for image_path in train_dogs:
        images.append(misc.imread(image_path, mode='RGB'))
    X = np.asarray(images)
    #print(images[1].dtype)
    print('Imported train: {0} cats and {1} dogs.'.format(len(train_cats), len(train_dogs)))
    return X, y

def import_test_images():
    test_path = [os.path.join(TEST_DIR, str(x[1])+'.jpg') for x in IDS_TEST_EXAMPLES]
    y = [int(x[0]=='cat') for x in IDS_TEST_EXAMPLES]
    images = []
    for image_path in test_path:
        images.append(misc.imread(image_path, mode='RGB'))
    X = np.asarray(images)
    print('Imported test: {0}.'.format(len(images)))
    return X, y

def print_images(X, title =''):
    fig, ax = plt.subplots()
    ax.imshow(X)
    ax.set_title(title)
    plt.show()


X_train, y_train = import_train_images()
X_test, y_test = import_test_images()

# print examples
print_images(X_train[1], 'Train Cat Example')
print_images(X_train[-1], 'Train Dog Example')
print_images(X_test[1], 'Test Example')

# save a nice cat that wants to becomes a dog
#x_0_origin = X[4,:].squeeze()
#x_0 = x_0_origin.reshape(1, -1)
#y_0 = y[4,:].squeeze()
#print_images(x_0_origin)

# image2vector train
X_train = X_train.reshape(X_train.shape[0], -1).squeeze()
y_train = y_train.T

# we don't train a statsmodels GLM, because data are too heavy for its implentation.

# sklearn LR with L2 regularization
# search for the best C hyperparameter
lr_l2_CV = linear_model.LogisticRegressionCV(penalty = 'l2', random_state = 42)
lr_l2_CV.fit(X_train, y_train)
bestC = lr_l2_CV.C_[0]
print('Best C found: {0}'.format(bestC))
del lr_l2_CV

# retrain LR with the best C
# this is the same than above, but currently adversarialLogistic 
# doesn't support linear_model.LogisticRegressionCV
lr_l2 = linear_model.LogisticRegression(penalty = 'l2', random_state = 42, C=bestC)
lr_l2.fit(X_train, y_train)


# Perturbate the cat power

adv = AdversarialLogistic(lr_l2, lower_bound=0, upper_bound=255)
print(adv.beta_hat.shape)
# WARNING: heavy on RAM
#adv.compute_covariance(X_train, y_train)

# save adv for latter reuse:
#with open('adv.pkl', 'wb') as output:
#    pickle.dump(adv, output, pickle.HIGHEST_PROTOCOL)

# load previously saved adv
with open('adv.pkl', 'rb') as input:
    adv = pickle.load(input)


#adv.plot_lambda_vs_alpha(x_0)



for index, test_image in enumerate(X_test):
    print('x_0 is predicted as: {0}'.format(lr_l2.predict(x_0)[0]))
    x_0 = test_image.reshape(test_image.shape[0], -1).squeeze()
    y_0 = y_test[index]
    pred_x_0 = lr_l2.predict(x_0)
    if (pred_x_0 != y_test[index]):
        print('Test example #{0} is not predicted correctly by the model({1} vs {2}). Ignored.'.format(index, pred_x_0, y_0))
        continue
    x_adv = [adv.compute_adversarial_perturbation(x_0, y_0, alpha=alpha, out_bounds='clipping') for alpha in ALPHAS]

    f, axarr = plt.subplots(1+len(ALPHAS),2)
    axarr[0,0].imshow(x_0.reshape(64,64,3))
    axarr[0,0].set_title('Original Example')
    axarr[0,1].imshow(x_adv[0]['x_adv_0'][1:].reshape(64,64,3))
    axarr[0,1].set_title('Orthogonal Projection (α = 0.5)')
    for i, alpha in enumerate(ALPHAS):
        axarr[1+i,0].imshow(x_adv[i]['x_adv_star'][1:].reshape(64,64,3))
        axarr[1+i,0].set_title('Adversarial Example (α = {0})'.format(alpha))
        delta_star_plot = np.abs(x_adv[i]['x_adv_star'][1:].reshape(64,64,3) - x_0.reshape(64,64,3))
        axarr[1+i,1].imshow(delta_star_plot)
        axarr[1+i,1].set_title('Adversarial Perturbation (α = {0})'.format(alpha))
    plt.show()
