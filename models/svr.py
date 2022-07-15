import numpy as np
import pandas as pd
from sklearn import linear_model
# from sklearn.model_selection import GroupKFold
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import time



param_grid = {}
optimal_paramaters = []

#Precalculate fingerprints in pickle files

fingerprints = np.load('fingerprints.npy')
scores = np.load('scores.npy')
scores = scores.astype('float64')

# fingerprints = fingerprints[0:1000]
# scores = scores[0:1000]
checkpoint = time.time()

# Linear Regression Model

model = svm.SVR()

scoring = ['r2', 'neg_root_mean_squared_error']
# splitter = GroupKFold(n_splits=3)
# split_iterator = splitter.split(fingerprints, scores, scaffold_groups)
results = cross_validate(model, fingerprints, scores, cv=3, scoring=scoring)

crossval_means = {}

for x, y in results.items():
    crossval_means[x] = np.mean(y)

crossval_means['r2_stdev'] = np.std(results['test_r2'])

print("SVR results")
print(crossval_means)
print(f'\n\nModel trained in {(time.time() - checkpoint):.1f} seconds', flush=True)
