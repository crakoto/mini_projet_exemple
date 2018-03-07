#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: isabelleguyon

This is an example of program that tests the Iris challenge Preprocessor class.
Another style is to incorporate the test as a main function in the Data manager class itself.
"""
from sys import path
path.append ("../ingestion_program") # Contains libraries you will need
from data_manager import DataManager  # such as DataManager

from prepro import Preprocessor
input_dir = "../sample_data"
output_dir = "../resuts"

basename = 'Iris'
D = DataManager(basename, input_dir) # Load data
print("*** Original data ***")
print D

Prepro = Preprocessor()
 
# Preprocess on the data and load it back into D
D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
D.data['X_test'] = Prepro.transform(D.data['X_test'])
  
# Here show something that proves that the preprocessing worked fine
print("*** Transformed data ***")
print D

# Preprocessing gives you opportunities of visualization:
# Scatter-plots of the 2 first principal components
# Scatter plots of pairs of features that are most relevant
import matplotlib.pyplot as plt
X = D.data['X_train']
Y = D.data['Y_train']
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()