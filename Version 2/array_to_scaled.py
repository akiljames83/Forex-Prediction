'''
- take in current value
- take in next 30 values
- fit then next 30 to the pc of the current
- output is the scaled values
- y value is the scaled value compared to the previous values
'''
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import pickle
import time
import os

#########################
## Import Pickled Data ##
#########################

matrix = pickle.load(open("four_sub_categories_600k.pk", "rb"))

arr1 = matrix.pop(-1)
arr2 = matrix.pop(-1)
arr3 = matrix.pop(-1)
arr4 = matrix.pop(-1)
arr_validation = matrix.pop(-1)

####################
## Main Functions ##
####################

def percent_change_and_scale(arr):
	scaler = MinMaxScaler()
	pc_data = list()
	for i in arr[1:-1]:
		val = (i - arr[0])/arr[0]
		pc_data.append(val)

	pc_last = (arr[-1] - arr[0])/arr[0]
	# takes in a Matrix of N values (all 1x1), so reshape is necessary for transfrom
	scaled_data = scaler.fit_transform(np.array(pc_data).reshape(-1,1))
	scaled_data = scaled_data.reshape(-1)
	scaled_last = float(scaler.transform(pc_last))
	return scaled_data, scaled_last

def processing_input_array(array, seq_length=30):
	processed_array=[]
	length = len(array)
	for i in range(length - seq_length - 2):
		arr = array[i:i+seq_length+1]
		# len(Seq Arr) = SL -1, since, first data is used for pc not prediction
		scaled_arr, scaled_target = percent_change_and_scale(arr)
		processed_array.append([scaled_arr,scaled_target])
		# if i == 4: break
	return processed_array

# Storage array for: 
processed_array = [] # trained
validation_array = [] # validation

######################################
## Processing Train and Val. Groups ##
######################################

for curr_array in [arr1, arr2, arr3, arr4]:
	processed_array.append(processing_input_array(curr_array))

validation_array = processing_input_array(arr_validation)

_ = []
for i in processed_array:
	_ = _ + i
processed_array = _

print("Length of processed array is {}.\n".format(len(processed_array)))

###################
### Extra Stuff ###
###################

random.shuffle(processed_array)
random.shuffle(validation_array)

##################################
### Process Train & Validation ###
##################################

train_X = [] 
train_y = []
validation_X = [] 
validation_y = []

for seq, target in processed_array:
	train_X.append(seq)
	train_y.append(target)

for seq, target in validation_array:
	validation_X.append(seq)
	validation_y.append(target)

train_X = np.array(train_X)
validation_X = np.array(validation_X)

print(train_X[0][:5],"\t",type(train_X[0]))
print(train_y[:5],"\t",type(train_y[0]),"\n")

print(validation_X[0][:5],"\t",type(validation_X[0]))
print(validation_y[:5],"\t",type(validation_y[0]))

pickle.dump(train_X, open("pickles/train_X.pk","wb"))
pickle.dump(train_y, open("pickles/train_Y.pk","wb"))

pickle.dump(validation_X, open("pickles/validation_X.pk","wb"))
pickle.dump(validation_y, open("pickles/validation_Y.pk","wb"))
