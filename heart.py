# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 00:03:29 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('heart.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , 13].values

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder_1 = OneHotEncoder(categorical_features = [3])
X = onehotencoder_1.fit_transform(X).toarray()

onehotencoder_2 = OneHotEncoder(categorical_features = [9])
X = onehotencoder_2.fit_transform(X).toarray()

onehotencoder_3 = OneHotEncoder(categorical_features = [11])
X = onehotencoder_3.fit_transform(X).toarray()

onehotencoder_4 = OneHotEncoder(categorical_features = [15])
X = onehotencoder_4.fit_transform(X).toarray()

onehotencoder_5 = OneHotEncoder(categorical_features = [18])
X = onehotencoder_5.fit_transform(X).toarray()

onehotencoder_6 = OneHotEncoder(categorical_features = [22])
X = onehotencoder_6.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.15 , random_state = 0)
 
import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

classifier = Sequential()
classifier.add(Dense(units = 13 , kernel_initializer = 'uniform' , activation = 'relu' , input_dim = 26))
classifier.add(Dense(units = 13 , kernel_initializer = 'uniform' , activation = 'relu'))
classifier.add(Dense(units = 13 , kernel_initializer = 'uniform' , activation = 'relu'))
classifier.add(Dense(units = 1 , kernel_initializer = 'uniform' , activation = 'sigmoid'))


classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

classifier.fit(X_train , y_train , batch_size = 4 , epochs = 100)

classifier.summary()

y_pred = classifier.predict(X_test)
y_pred1 = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred1)
