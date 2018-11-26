import matplotlib.pyplot as plt
import pickle
import time

with open('real_prices.pk', 'rb') as f1:
	real = pickle.load(f1)
with open('forecasted.pk', 'rb') as f2:
	forecasted = pickle.load(f2)

fig = plt.figure()
ax1 = fig.add_subplot(111)

if len(real) == len(forecasted):
	print('equivalent lengths')
	x = list(range(len(real)))
	ax1.scatter(x, real, s=10, c='b', marker="s", label='real')
	ax1.scatter(x,forecasted, s=10, c='r', marker="o", label='forecasted')
else:
	print("inequivalent lengths",len(forecasted), len(real))
	x1 = list(range(len(real)))
	x2 = list(range(len(forecasted)))
	ax1.scatter(x1, real, s=10, c='b', marker="s", label='real')
	ax1.scatter(x2,forecasted, s=10, c='r', marker="o", label='forecasted')

trend_real = real[-1] - real[0]
trend_pred = forecasted[-1] - forecasted[0]
print("Model predicted market: {}.\nThe market actually moved {}.".format(trend_pred, trend_real))
input("Ready for Graph?")
plt.legend(loc='upper left');
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.savefig('images/BIDI-LSTM-16-4-{}.png'.format(str(time.time())[0:9]))
plt.show()