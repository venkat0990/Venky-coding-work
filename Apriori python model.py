# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:17:04 2019

@author: Don't Give up
"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset 
dataset = pd.read_csv('Market_Basket_Optimisation.csv' , header = None)
records = [] 
for i in range(0, 7501):
    records.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Train Apriori Model
from apyori import apriori
rules = apriori(records, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
print(results)