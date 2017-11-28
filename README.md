
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

```
python preprocess_dogscats.py
```

### 2.3 Train Model & Compute Adversarial Examples

```
mkdir -p images/cats/test
mkdir -p images/cats/test2
mkdir -p obj/x_adv
python cat_non-cat.py >>log_cats.txt 2>&1
```

