# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 18:21:22 2019

@author: Don't Give up
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:3].values

# Fitting Decision Tree Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 280, random_state = 0)
regressor.fit(X, Y)

# Predicting a new result
Y_pred = regressor.predict([[6.5]])

# Visualizing the second Decision Tree Regression results
X_grid = np. arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color = 'green')
plt.plot(X_grid, regressor.predict(X_grid), color = 'brown')
plt.title('Truth or Bluff(Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
