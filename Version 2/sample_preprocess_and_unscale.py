from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time

def scaled_to_true(scaler, data, initial):
	# invert the scale
	pc = scaler.inverse_transform(data)

	# invert the percent change
	y = (pc[0, 0] + 1)*initial
	return y

def percent_change_and_scale(arr):
	'''
	array of data including the future predicted
	outputs: scaled values, pc of last
	'''
	scaler = MinMaxScaler()
	pc_data = list()
	for i in arr[1:-1]:
		val = (i - arr[0])/arr[0]
		pc_data.append(val)

	pc_last = (arr[-1] - arr[0])/arr[0]

	scaled_data = scaler.fit_transform(np.array(pc_data).reshape(-1,1))
	print(scaled_data.reshape(-1), type(scaled_data))
	scaled_last = scaler.transform(pc_last)
	print(type(scaled_last[0]))
	return scaled_data, scaled_last, scaler, arr[0]

t0 = time.time()
array = [1.4933,1.4930,1.4928,1.4923,1.4932,1.4935]

scaled_data, scaled_last, scaler, initial = percent_change_and_scale(array)

# would predict some random value using the model
predicted = np.array([0.8]).reshape(-1, 1)

y = scaled_to_true(scaler, scaled_last, initial)
yhat = scaled_to_true(scaler, predicted, initial)
f = time.time()
print("\n\nPredicted: {} Actual: {}".format(yhat, y))

print("Time taken is: {}".format(f-t0))
