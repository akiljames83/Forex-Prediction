import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from mpl_finance import candlestick_ohlc
import pickle
import time
# def convert_time(time):
# 	hour = time[0:2]
# 	minute = time[2:4]
# 	second = time[4:6]
# 	return ":".join([hour,minute,second])

# def convert_date(date):
# 	MONTH = {"01":"January","02":"February","03":"March","04":"April","05":"May","06":"June","07":"July","08":"August","09":"September","10":"October","11":"November","12":"December"}
# 	year = date[0:4]
# 	month = date[4:6]
# 	day = date[6:8]
# 	return "-".join([MONTH[month],day,year])
# start = t.time()
# with open("USDCAD.txt","r") as f:
# 	file = f.readlines()
# 	data = []
# 	for index, fil in enumerate(file[555000:]):
# 		if index < 10000:
# 			data.append(fil[7:])
# 		else: break

# date, time, openp, high, low, close, vol = np.loadtxt(data,delimiter=',',unpack=True)

# stuff = [date, time, openp, high, low, close, vol]


# x = 0
# y = len(date)
# ohlc = []

# while x < y: #float((str(date[x]) + str(time[x]))[:-2])
# 	append_me =  x,openp[x], high[x], low[x], close[x], vol[x]
# 	ohlc.append(append_me)
# 	x += 1
# #print(ohlc)

a = time.time()
ohlc = pickle.load(open("tick_ohlc_data.pickle", "rb"))
print("Loading in the data took",time.time() - a,"seconds.")




fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))
candlestick_ohlc(ax1,ohlc[-5000:])
plt.show()
