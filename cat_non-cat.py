"""
Dogs vs Cats

This script runs a logistic regression to classify images as cat or dog, 
and computes adversarial images according to different confidence levels.

Data and details are available at: https://www.kaggle.com/c/dogs-vs-cats/data
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import misc
import os
import glob
import pickle
from adversarialLogistic import AdversarialLogistic
from adversarialLogistic import plot_intensity_vs_level

TRAIN_DIR = 'data/cats/data64/train'
TEST_DIR = 'data/cats/data64/test'
# cats and dogs who want to change their identities from TEST_DIR:
# squared images without gray bands
IDS_TEST2_EXAMPLES = [('dog', 2), ('cat', 5), ('dog', 21), ('cat', 28), ('cat', 45), ('cat', 58), ('cat', 90)]
# values of alphas to compute adversarial examples
ALPHAS = [0.75, 0.9, 0.95]
COLORS_MODELS = ['orchid']
DEBUG = False

def print_image(X, title =''):
    fig, ax = plt.subplots()
    ax.imshow(X)
    ax.set_title(title)
    plt.show()

def image2vector(x):
    return x.reshape(x.shape[0], -1).squeeze()

def vector2image(x):
    if x.shape[-1] == 64*64*3+1:
        # contains the constant
        assert(x[0]==1.0)
        return x[1:].reshape(-1, 64, 64, 3).squeeze().astype(np.uint8)
    else:
        return x.reshape(-1, 64, 64, 3).squeeze().astype(np.uint8)


def import_train_images():
    train_cats = sorted(glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg')))
    train_dogs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'dog*.jpg')))
    if DEBUG:
        train_cats = train_cats[0:100]
        train_dogs = train_dogs[0:100]
    n = len(train_cats) + len(train_dogs)
    y = np.zeros((1,n))
    images = []
    for image_path in train_cats:
        images.append(misc.imread(image_path, mode='RGB'))
    y[0, 0:len(train_cats)] = 1
    for image_path in train_dogs:
        images.append(misc.imread(image_path, mode='RGB'))
    X = np.asarray(images)
    # image2vector
    X = image2vector(X)
    y = y.T
    print('Imported train: {0} cats and {1} dogs.'.format(len(train_cats), len(train_dogs)))
    return X, y

def import_test_images():
    test_path = [os.path.join(TEST_DIR, str(x[1])+'.jpg') for x in IDS_TEST2_EXAMPLES]
    y = [int(x[0]=='cat') for x in IDS_TEST2_EXAMPLES]
    y = np.matrix(y)
    images = []
    for image_path in test_path:
        images.append(misc.imread(image_path, mode='RGB'))
    X = np.array(images)
    # image2vector
    X = image2vector(X)
    y = y.T
    print('Imported test: {0} images.'.format(len(images)))
    return X, y

def save_obj(x, filename = 'obj.pkl'):
    with open(filename, 'wb') as output:
        pickle.dump(x, output, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    with open(filename, 'rb') as input:
        adv = pickle.load(input)
    return adv

def x_adv_list2png(x_0, x_adv_list, filename):
    """
    Save an image containing all the adversarial examples in x_adv_list.
    """
    if len(x_adv_list) == 0:
        return
    f, axarr = plt.subplots(1+len(x_adv_list),2, figsize=(7, 9), dpi=150)
    axarr[0,0].imshow(vector2image(x_0))
    axarr[0,0].set_title('Original Example')
    axarr[0,1].imshow(vector2image(x_adv_list[0]['x_adv_0']))
    axarr[0,1].set_title('Orthogonal Projection\nα = 0.5')
    for i, x_adv in enumerate(x_adv_list):
        alpha = x_adv['alpha']
        axarr[1+i,0].imshow(vector2image(x_adv['x_adv_star']))
        axarr[1+i,0].set_title('Adversarial Example\nα = {0}, δ = {1:.2f}'.format(alpha, x_adv['lambda_star']))
        delta_star_plot = vector2image(100*np.abs(x_adv['x_adv_star'][1:] - x_0))
        axarr[1+i,1].imshow(delta_star_plot)
        axarr[1+i,1].set_title('Adversarial Perturbation\nα = {0}, x100'.format(alpha))
    f.tight_layout()
    plt.savefig(filename)
    plt.close()


#--------------------------------
# I - Data
#--------------------------------

# TRAIN_DIR images will be split between train and test
X, y = import_train_images()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
del X, y # avoid bugs

# Only a few TEST_DIR images will be used for illustration
X_test2, y_test2 = import_test_images()

# print examples
if DEBUG:
    print_image(vector2image(X_train[1,:]), 'Train Cat Example')
    print_image(vector2image(X_train[0,:]), 'Train Dog Example')
    print_image(vector2image(X_test[0,:]), 'Test Example')
    print_image(vector2image(X_test2[0,:]), 'Test2 Example')


#--------------------------------
# II - Train Logistic Regression
#--------------------------------

# we don't train a statsmodels GLM, because data are too heavy for its implentation.

if not os.path.isfile('obj/adv.pkl'):
    # sklearn LR with L2 regularization
    # search for the best C hyperparameter
    lr_l2_CV = linear_model.LogisticRegressionCV(penalty = 'l2', solver='sag', Cs=100,
        random_state = 42, n_jobs=-1)
    # Cs=100 : grid of 100 values
    lr_l2_CV.fit(X_train, y_train)
    bestC = lr_l2_CV.C_[0]
    print('Best C found: {0}'.format(bestC))
    del lr_l2_CV

    # retrain LR with the best C
    # this is the same than above, but currently adversarialLogistic 
    # doesn't support linear_model.LogisticRegressionCV
    lr_l2 = linear_model.LogisticRegression(penalty = 'l2', solver='sag', random_state = 42, C=bestC, n_jobs=-1)
    lr_l2.fit(X_train, y_train)

    lr_l2_acc_is = lr_l2.score(X = X_train, y = y_train)
    lr_l2_acc_oos = lr_l2.score(X = X_test, y = y_test)
    print('Accuracy in-sample: {0}'.format(lr_l2_acc_is))
    print('Accuracy out-of-sample: {0}'.format(lr_l2_acc_oos))


#------------------------------------
# III - Prepare AdversarialLogistic 
#------------------------------------

# Perturbate the cat power

if os.path.isfile('obj/adv.pkl'):
    # load previously saved adv
    adv = load_obj('obj/adv.pkl')
    lr_l2 = adv.model
else:
    adv = AdversarialLogistic(lr_l2, lower_bound=0, upper_bound=255)
    
    # WARNING: heavy on RAM
    adv.compute_covariance(X_train, y_train)

    # save adv for latter reuse:
    save_obj(adv, filename = 'obj/adv.pkl')


#---------------------------------------------
# IV - Compute Adversarial Images for X_test2 
#---------------------------------------------

# Used to plot for intensity VS level

print('Compute Adversarial Images for X_test2...')

alphas_list = np.arange(0.001, 0.999, 0.001).tolist()

for index, x_0 in enumerate(X_test2):
    x_0 = x_0.reshape(1, -1) # shape: (12288,) -> (12288,1)
    y_0 = y_test[index].squeeze()
    pred_x_0 = lr_l2.predict(x_0)

    print('Test2 example #{0} (y={1}) predicted as: {2}'.format(index, y_0, pred_x_0[0]))
    # save adversarial images
    x_adv_list = []
    for alpha in ALPHAS:
        x_adv = adv.compute_adversarial_perturbation(x_0, y_0, alpha=alpha, out_bounds='clipping')
        x_adv_list.append(x_adv)

    # save x_adv_list
    save_obj(x_adv_list, filename = 'obj/x_adv/test2_'+str(index)+'.pkl')

    # plot and save the images
    x_adv_list2png(x_0, x_adv_list, filename='images/cats/test2/adv_picture_'+str(index)+'.png')

    # plot intensity versus level
    x_adv_list = []
    for alpha in alphas_list:
        x_adv = adv.compute_adversarial_perturbation(x_0, y_0, alpha=alpha, 
            out_bounds='clipping', verbose_bounds=False)
        x_adv_list.append(x_adv)

    plot_intensity_vs_level(x_adv_list, labels = None,
        colors = COLORS_MODELS, filename='images/cats_intensity_level_x_test2_'+str(index)+'.png')

del alphas_list


#-------------------------------------------
# V - Compute Adversarial Images for X_test 
#-------------------------------------------

# Used to plot the density of intensities in the X_test population, for different levels

lambds_list = [] # list of list each containing the values of lambdas associated with alphas_list
alphas_list = []

print('Compute Adversarial Images for X_test...')
for index, x_0 in enumerate(X_test):
    x_0 = x_0.reshape(1, -1) # shape: (12288,) -> (12288,1)
    y_0 = y_test[index].squeeze()
    pred_x_0 = lr_l2.predict(x_0)

    x_adv_list = []
    for alpha in ALPHAS:
        try:
            x_adv = adv.compute_adversarial_perturbation(x_0, y_0, alpha=alpha, out_bounds='clipping')
            x_adv_list.append(x_adv)
            lambds_list.append(x_adv['lambda_star'])
            alphas_list.append(alpha)
        except ArithmeticError:
            print('Underflow. Skipping computation of test example #{0}.'.format(index))
            continue

    # save x_adv_list
    save_obj(x_adv_list, filename = 'obj/x_adv/test_'+str(index)+'.pkl')

    print('Test example #{0} (y={1}) predicted as: {2}'.format(index, y_0, pred_x_0[0]))
    # plot and save the images
    x_adv_list2png(x_0, x_adv_list, filename='images/cats/test/adv_picture_'+str(index)+'.png')


# plot the distribution of lamdbas with respect to alphas
df_lambds = pd.DataFrame({'alpha': alphas_list, 'lambd': lambds_list})
save_obj(df_lambds, filename = 'obj/df_lambds.pkl')

fig = plt.figure(figsize=(7, 5), dpi=150)
sns.violinplot(x=df_lambds["alpha"], y=df_lambds["lambd"], palette="Blues", gridsize=2000, scale_hue=False, saturation=0.9)
plt.xlabel('Misclassification level (α)')
plt.ylabel('Intensity of the pertubation (λ)')
plt.savefig('images/cats_violinplot.png')
plt.ylim(-60, 70)
plt.savefig('images/cats_violinplot_zoom.png')
plt.savefig('images/cats_violinplot_zoom.pdf')
