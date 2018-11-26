from pandas import read_csv, DataFrame, concat, Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from math import sqrt, floor, log
import numpy as np
import pickle

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a statistically stationary series of data
def difference(dataset, interval=1):
	'''
	Create a time series is one whose statistical properties such as mean, variance, autocorrelation, etc.
	are all constant over time.
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
def scale(scraped):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(scraped)
	# transform train
	scraped = scraped.reshape(scraped.shape[0], scraped.shape[1])
	scraped_scaled = scaler.transform(scraped)
	return scaler, scraped_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# take in X data for a prediction
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

#####################
## Start Modelling ##
#####################

# Take in old data
series = read_csv('updated-scrape.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)
 
# transform data to be prepared for supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
scraped_data = supervised.values

# transform the scale of the data
scaler, scraped_scaled = scale(scraped_data)

# walk-forward validation on the test data
predictions = list()

model = load_model("Bidirectional-LSTM-D50p-16-4.keras")
for i in range(len(scraped_scaled)):
	# make one-step forecast
	X, y = scraped_scaled[i, 0:-1], scraped_scaled[i, -1]
	yhat = forecast_lstm(model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(scraped_scaled)+1-i)
	# store forecast
	predictions.append(yhat)

# Length of time recorded + X time steps forward
extended = predictions
for i in range(60+2):
	X = np.array([extended[-2]])
	y = extended[-1]
	yhat = forecast_lstm(model, 1, X)
	yhat = invert_scale(scaler, X, yhat)
	yhat = inverse_difference(raw_values, yhat, len(scraped_scaled)+1-i)
	extended.append(yhat)

# Store the predictions in an array so that the predictions can be graphed and compared with real-time data
pickle.dump(extended, open("forecasted.pk","wb"))