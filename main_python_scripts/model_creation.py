#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Keras Model and Premaking the model
'''
from pandas import read_csv, DataFrame, concat, Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt, floor, log
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
import numpy as np
from time import time

###############
## FUNCTIONS ##
###############

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	'''
	Statistical Stationary: 
	Definition: A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time.
	'''
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	#model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Bidirectional(LSTM(16, stateful=True, return_sequences=True),batch_input_shape=(batch_size, X.shape[1], X.shape[2])))
	model.add(Bidirectional(LSTM(neurons)))
	model.add(Dense(1))
	model.compile(loss='binary_crossentropy', optimizer='adam',metrics=["accuracy"])
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False,validation_split=0.1)
		model.reset_states()
		print("Epoch %d finished." % i)
	model.save("Bidirectional-LSTM-32-4.keras")
	return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def preprocessing_in(csvfile='USDCAD.csv'):
	# Load in the CSV Data
	series = read_csv('USDCAD.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

	# transform data to be stationary
	raw_values = series.values
	diff_values = difference(raw_values, 1)
	 
	# transform data to be supervised learning
	supervised = timeseries_to_supervised(diff_values, 1)
	supervised_values = supervised.values
	
	# Find the value of the index for the data split: 5% split
	split = floor(len(supervised_values)*0.2)*-1

	# split data into train and test-sets
	train, test = supervised_values[0:split], supervised_values[split:]
	
	print("Length of Train Dataset: %d.\nLength of Test Dataset: %d.\n" % (len(train), len(test)))

	# transform the scale of the data
	scaler, train_scaled, test_scaled = scale(train, test)

	return train, scaler, train_scaled, test_scaled, raw_values, split

def pip_weighted_error(actual, predicted):
	'''
	Actual value is a numpy array
	predicition values is a python list
	'''
	total = 0
	assert(len(actual) == len(predicted))
	for i in range(len(actual)):
		dif = round(actual[i] - round(predicted[i],4),4)
		if dif == 0:
			total += 1
			#print(1)
			continue
		print("Predicted:%f\tActual:%f" % (actual[i], round(predicted[i],4)))
		alog = abs(log(dif**2))
		print(0.98 - (18.43 - alog)/18.43)
		total += 0.98 - (18.43 - alog)/18.43
		#print(1 - (8 - alog)/8)
	total /= len(actual)
	return total
#####################
### PREPROCESSING ###
#####################

train, scaler, train_scaled, test_scaled, raw_values, split = preprocessing_in('USDCAD.csv')

#############
### MODEL ###
#############
 
# fit the model
lstm_model = fit_lstm(train_scaled, 1, 30, 4)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

print("Training finished.\n")

###################
### PREDICTIONS ###
###################

# walk-forward validation on the test data
predictions = list()
times = []
#print("\n\n",test_scaled,"\n\n")
for i in range(len(test_scaled)):
	# make one-step forecast
	start = time()
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	if i == 8:
		print("__________________")
		print(X,type(X),y,type(y))
	yhat = forecast_lstm(lstm_model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train) + i + 1]
	times.append(time()-start)
	print('Predicted=%f, Expected=%f' % (yhat, expected))
extended = predictions[-2:]

for i in range(30):
	X = np.array([extended[-2]])
	y = extended[-1]
	yhat = forecast_lstm(lstm_model, 1, X)
	yhat = invert_scale(scaler, X, yhat)
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	extended.append(yhat)

print(extended[1:])
print("Pip weighted error: %.3f" % pip_weighted_error(raw_values[split:], predictions))
