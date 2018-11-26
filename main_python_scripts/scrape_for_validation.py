import os
import time
import numpy as np
import json
import pickle

price = []
for i in range(120):
	os.system('curl "https://forex.1forge.com/1.0.3/quotes?pairs=USDCAD&api_key=iC6MdJbRlx3gsTxiW7Fp84Is3hl6KNgP" > stdout.txt 2>&1') # python truefx.py
	# curl "https://forex.1forge.com/1.0.3/quotes?pairs=USDCAD&api_key=iC6MdJbRlx3gsTxiW7Fp84Is3hl6KNgP" >> stdout.txt 2>&1
	# clean up text file// or change to single > to overwrite the file
	with open("stdout.txt","r") as sysout:
		a = sysout.readlines()[-1][1:-1]
		print(a)
		# forex dictionary for the timestamp, bid and ask
		forex_dict = json.loads(a)
		#timestamp = time.strftime('%Y-%m-%dT%H:%M:%S %Z',time.localtime(forex_dict["timestamp"]))
	
	price.append(forex_dict["price"])
	time.sleep(60) # 15 minute break = 60*15

pickle.dump(price, open("real_prices.pk","wb"))