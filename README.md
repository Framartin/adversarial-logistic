
# Intensity of Adversarial Examples for Logistic Regression

TODO: description

## 0. Installation

```
#sudo pip3 install virtualenv python3-tk
virtualenv -p python3 venv
. venv/bin/activate
pip install -r requirements.txt
```

For an installation on a server without Xwindows, you can add `backend : Agg` to `.config/matplotlib/matplotlibrc`. 


## 1. Spam

TODO


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
- make use of `adversarialLogistic.py` to compute adversarial perturbations to the missclassification levels 0.75, 0.9 and 0.95
- draw plots of delta versus alpha for a small sample of squared images of `data/cats/data64/test`, called `test2`
- draw a violinplot of the deltas computed on the labelled test set (30% of `data/cats/data64/train`, called `test`)

**Becareful:** the computation of the variance-covariance needs quite a lot of RAM. 8 Gio should be enough.

```
mkdir -p images/cats/test
mkdir -p images/cats/test2
mkdir -p obj/x_adv
python cat_non-cat.py >>log_cats.txt 2>&1
```

The following informations are saved in the log file:

- in-sample and out-of-sample accuracies
- number of skipped examples due to underflow ($a<10^{-7}$ in the 2nd degree eq. solver). The detection of underflow should be improve in the future.
- value of C, the regularization parameter, choosen by CV on the train set
- for each example, the predicted class and the real one 

```
head log_cats.txt
cat log_cats.txt | grep "Underflow"
```
