import numpy as np
import pickle
from time import time as t
a = t()
with open("USDCAD.txt","r") as f:
	file = f.readlines()
	data = []
	for index, fil in enumerate(file[1:]):
			data.append(fil[7:])

date, time, openp, high, low, close, vol = np.loadtxt(data,delimiter=',',unpack=True)

stuff = [date, time, openp, high, low, close, vol]


x = 0
y = len(date)
ohlc = []

while x < y: #float((str(date[x]) + str(time[x]))[:-2])
	append_me =  x,openp[x], high[x], low[x], close[x], vol[x]
	ohlc.append(append_me)
	x += 1

pickle.dump(ohlc, open("tick_ohlc_data.pickle","wb"))
print(a - t())