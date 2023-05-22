import tensorflow as tf
import shutil
from zipfile import ZipFile as zf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.feature_selection import SelectKBest,chi2,RFE,RFECV
from sklearn.model_selection import train_test_split,cross_validate,RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor
from keras.layers import Dense,Flatten,Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras_tuner import HyperParameters,RandomSearch
# What version of Python do you have?
import sys
from keras import __version__
import pandas as pd
import sklearn as sk
'''
The following two 'detect_outliers' functions have been taken from 'Finding an outlier in a dataset using Python' by Krish Naik. All credit goes to him.
https://www.youtube.com/embed/rzR_cKnkD18
'''
outliers = []

def detect_outliers_std(col):
    threshold=3 # Can change
    mean = np.mean(col)
    std = np.std(col)

    for point in col:
        z_score = (point-mean)/std
        if np.abs(z_score) > threshold:
            outliers.append(point)
        
    return outliers
def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

def my_fun_detect_outliers_combined(col):
    iqr = sorted(detect_outliers_iqr(col))
    std = sorted(detect_outliers_std(col))
    return std,iqr

def calculate_correlation(df, column1, column2):
    df[column2] = OrdinalEncoder().fit_transform(df[column2].to_numpy().reshape(-1,1))
    correlation = np.around(df[column1].corr(df[column2]),4)
    return correlation
def convert_days(days):
    years, days = divmod(days, 365) 
    months, days = divmod(days, 30)  
    print(f'Most extreme patient spent {years} years, {months} months and {days} days in the clinic')
def clean_useless_cols(df, y_label):
    print('#########################')
    print(f'starting with {df.columns.size} cols')
    print('#########################')
    print(df.corr().nlargest(df.columns.size, y_label)
          [y_label].sort_values(ascending=False))
    irrelevant_cols = pd.DataFrame(df.corr()[y_label]).isna()
    ir_cols_list = irrelevant_cols[irrelevant_cols[y_label]].index
    print('\n#########################')
    print('irrelevant cols found (constants):')
    print('#########################')
    print(df[ir_cols_list].nunique())
    df.drop(ir_cols_list, axis=1, inplace=True)
    print('\n#########################')
    print(f'voila! set cleared. {df.columns.size} cols left to work with')
    print('#########################')

'''
The following function 'detect_software_version_and_GPU' (below) is not mine, credit goes to: Professor Jeff Heaton
https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/tensorflow-install-march-2023.ipynb
'''
def detect_software_version_and_GPU():
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {__version__}")
    print()
    print(f"Python {sys.version}")
    print(f"Pandas {pd.__version__}")
    print(f"Scikit-Learn {sk.__version__}")
    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")