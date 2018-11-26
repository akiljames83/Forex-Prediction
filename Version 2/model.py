import tensorflow as tf
from tensorflow.keras.layers import LSTM, CuDNNLSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1
import numpy as np
import pickle
import time

NUM_EPOCHS = 10
BATCH_SIZE = 128
PERCENT_REDUC_VAL = .2
NAME = "LSTM-16-4-1-TimeStamp:{}".format(time.time())

X_train = pickle.load(open("pickles/train_X.pk", "rb"))
y_train = pickle.load(open("pickles/train_Y.pk", "rb"))

val_X = pickle.load(open("pickles/validation_X.pk", "rb"))
val_Y = pickle.load(open("pickles/validation_Y.pk", "rb"))

val_X = val_X[:int(len(val_X)*PERCENT_REDUC_VAL)]
val_Y = val_Y[:int(len(val_Y)*PERCENT_REDUC_VAL)]

X_train = np.expand_dims(X_train, axis=1)
val_X = np.expand_dims(val_X, axis=1)

print(val_X.shape)
print(X_train.shape)



model = Sequential()

#sentdex had this instead:
model.add(CuDNNLSTM(128,input_shape=(X_train.shape[1:]), 
				kernel_regularizer=l2(0.01),
                activity_regularizer=l1(0.01),
                return_sequences=False))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# model.add(CuDNNLSTM(128, 
# 				kernel_regularizer=l2(0.01),
#                 activity_regularizer=l1(0.01),
#                 return_sequences=True))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# model.add(CuDNNLSTM(128, 
# 				kernel_regularizer=l2(0.01),
#                 activity_regularizer=l1(0.01),
#                 return_sequences=True))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

#model.add(CuDNNLSTM(128,input_shape=(X_train.shape[1:]), return_sequences=True)) # activation relu
model.add(Dense(32, activation="tanh")) 
model.add(Dense(1, activation="relu"))

model.compile(loss=sparse_categorical_crossentropy,
			optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999),
			metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME)) #>> tensorboard --logidr=logs/


model.fit(
	X_train,
	y_train,
	epochs=NUM_EPOCHS,
	batch_size=BATCH_SIZE,
	validation_data=(val_X, val_Y))
model.save(("{}.keras").format(NAME))