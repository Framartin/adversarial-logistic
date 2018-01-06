
# Intensity of Adversarial Perturbations for Logistic Regression

This repository stores the code used in the article Martin Gubri (2018) "Adversarial Perturbation Intensity Strategy Achieving Chosen Intra-Technique Transferability Level for Logistic Regression" available [here](https://mg.frama.io/publication/intensity_adv_perturbation_logistic/).

The code provided can be used to compute adversarial examples that will be misclassified by a logistic regression to a chosen expected rate, under some conditions. The core of the code is `adversarialLogistic.py`, whereas 2 applications can be found in `spam.py` and `cat_non-cat.py`.


## 0. Installation

```
#sudo apt install virtualenv python3-tk
virtualenv -p python3 venv
. venv/bin/activate
pip install -r requirements.txt
```

For an installation on a server without Xwindows, you can add `backend : Agg` to `.config/matplotlib/matplotlibrc`. 


## 1. Spam

### 1.1 Download Data

```
mkdir -p data/spam
cd data/spam
wget https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.zip
unzip spambase.zip
cd ../..
```

### 1.2 Train Model & Compute Adversarial Examples

`spam.py` will:

- estimate 3 logistic regressions using statsmodels (GLM by IRLS), sklearn (L2-regularized and un-regularized)
- make use of `adversarialLogistic.py` to compute 3 adversarial perturbations, 1 for each model, to the misclassification level of 0.95
- draw a plot of lambda versus alpha for this example
- draw a violinplot of the lambdas computed on the all the test set for the 3 models

```
mkdir -p images/
python spam.py
```

## 2. Cats vs Dogs 

### 2.1 Download Data

Download `train.zip` and `test1.zip` on [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).

```
mkdir -p data/cats
unzip train.zip -d data/cats
unzip test1.zip -d data/cats
```

### 2.2 Preprocess Images

Resize images to 64x64 pixels, add gray bars if necessary and normalize the image luminance.

```
python preprocess_dogscats.py
```

The processed images are stored on `data/cats/data64`. You can safely delete `data/cats/train/`  and `data/cats/test1/`.

### 2.3 Train Model & Compute Adversarial Examples

`cat_non-cat.py` will:

- estimate a logistic regression using sklearn
- make use of `adversarialLogistic.py` to compute adversarial perturbations to the misclassification levels 0.75, 0.9 and 0.95
- draw plots of lambda versus alpha for a small sample of squared images of `data/cats/data64/test`, called `test2`
- draw a violinplot of the lambdas computed on the labeled test set (30% of `data/cats/data64/train`, called `test`)

**Becareful:** the computation of the variance-covariance matrix needs quite a lot of RAM. 8 Gio in total should be enough. The execution of the script takes a lot time, because computing the adversarial perturbations associated to each test examples is computationally expensive.

```
mkdir -p images/cats/test
mkdir -p images/cats/test2
mkdir -p obj/x_adv
python -u cat_non-cat.py 2>&1 | tee log_cats.txt
# -u argument is optional, but useful to keep track of the execution
```

The following informations are saved in the log file:

- in-sample and out-of-sample accuracies
- number of skipped examples due to underflow ($a<10^{-7}$ in the 2nd degree eq. solver). The detection of underflow should be improve in the future.
- value of C, the regularization parameter, chosen by CV on the train set
- for each example, the predicted class and the real one 

```
head log_cats.txt
cat log_cats.txt | grep "Underflow"
```
