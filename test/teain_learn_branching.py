from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

from joblib import dump, load

import pandas as pd
import numpy as np


# ------- Introduction ----------
# For training the Learned Branching model, with sample data
# -------- END -----------------



def read_and_split(fea_file, train_size = 0.6):
    df = pd.read_csv(fea_file)
    # score could have inf
    df = df.replace(np.inf, 1e8)
    label_col = ['score']
    features_col = ['s_01',
            's_03',
            's_04',
            's_05',
            's_06',
            's_07',
            's_08',
            's_11',
            's_12',
            'n_00',
            'n_01',
            'n_02',
            'n_03',
            'b_00',
            'b_01',
            'b_02',
            'b_03',
            'b_04',
            'b_05',
            'b_06',
            'b_07']
    
    X = df[features_col].values
    y = np.ravel(df[label_col].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = train_size, random_state=1)
    print('Train set: ', X_train.shape)
    print('Test  set: ', X_test.shape)
    return X_train, X_test, y_train, y_test, features_col



def train(fea_file, model, n_estimators=1000, split = 0.9):
    X_train, X_test, y_train, y_test,features_col = read_and_split(fea_file, train_size = split)
    params = {'n_estimators': n_estimators,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'random_state':0,
        'verbose':1,
          'loss': 'ls'}
    if model == 'gbdt':
        regr = GradientBoostingRegressor(**params)
    elif model == 'etr':
        regr = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=15, random_state=0)

    print('Train %s with %s ' %(fea_file, model))
    regr.fit(X_train, y_train)

    mdl_name = '%s_%s.joblib' %(model, fea_file.split('.')[0])
    dump(regr, mdl_name) 
    # predict
    y_test_pred = regr.predict(X_test)
    y_train_pred = regr.predict(X_train)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    print('mse_train: %.10f' %mse_train)
    print('mse_test : %.10f' %mse_test)
    print('feature_importances: \n')
    for i in range(len(features_col)):
        print('%s: %.8f'%(features_col[i], regr.feature_importances_[i]))
    
    if model == 'gbdt':
        test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
        for i, y_pred in enumerate(regr.staged_predict(X_test)):
            test_score[i] = regr.loss_(y_test, y_pred)
    
        fig = plt.figure(figsize=(6, 6))
        plt.subplot(1, 1, 1)
        plt.title('Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, regr.train_score_, 'b-',
                 label='Training Set Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
                 label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    train('fea_strong_1_0_20.csv', 'gbdt', n_estimators= 1000, split = 0.9)
    train('fea_strong_2_0_20.csv', 'gbdt', n_estimators= 1000, split = 0.9)
    train('fea_strong_3_0_20.csv', 'gbdt', n_estimators= 500, split = 0.9)
