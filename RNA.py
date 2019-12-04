###################################################################### IMPORTS

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

###################################################################### DATA PREPROCESSING

# Add names to the data just for further manipulations
names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'result']

# import our dataset from the csv file and assign the given names
data_set = pd.read_csv('DataOk.csv', names = names)

# get dataset records count
ds_size = data_set.shape[0]

# set the amount of data to be predicted
prediction = 100

# get the number of data for the training
training = ds_size - prediction

# get data to for the training in a new array. we remove the result tab because it's not used for the training
training_input = data_set.drop(columns = 'result').drop(data_set.index[training - prediction:ds_size - prediction ]).values
training_output = data_set.drop(columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']).drop(data_set.index[training - prediction:ds_size - prediction ]).values


training_output = np_utils.to_categorical(training_output, 4)

# data used as prediction input to get predicted output
prediction_input = data_set.drop(columns = 'result').drop(data_set.index[0:training]).values
prediction_output = data_set.drop(columns =['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']).drop(data_set.index[0:training]).values



###################################################################### RNA

# initialize the RNA
rna = Sequential()

# add the input layer 
rna.add(Dense(6, activation="sigmoid", input_shape=(6,)))

# add the output layer
rna.add(Dense(4, activation="softmax"))

# compile
rna.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

rna.fit(training_input, training_output, epochs=500, batch_size=32, verbose=0)



























