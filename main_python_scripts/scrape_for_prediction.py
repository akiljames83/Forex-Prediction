import os
import time
import numpy as np
import json
import pickle
import csv

spread = []
price = []
for i in range(60):
	os.system('curl "https://forex.1forge.com/1.0.3/quotes?pairs=USDCAD&api_key=iC6MdJbRlx3gsTxiW7Fp84Is3hl6KNgP" > stdout.txt 2>&1') # python truefx.py
	# curl "https://forex.1forge.com/1.0.3/quotes?pairs=USDCAD&api_key=iC6MdJbRlx3gsTxiW7Fp84Is3hl6KNgP" >> stdout.txt 2>&1
	# clean up text file// or change to single > to overwrite the file
	with open("stdout.txt","r") as sysout:
		a = sysout.readlines()[-1][1:-1]
		#print(a)
		# forex dictionary for the timestamp, bid and ask
		forex_dict = json.loads(a)
		timestamp = time.strftime('%Y-%m-%dT%H:%M:%S %Z',time.localtime(forex_dict["timestamp"]))
	price.append(forex_dict["price"])
	spread.append(forex_dict["ask"] - forex_dict["bid"])
	print(price[-1])
	time.sleep(60) # 15 minute break = 60*15

writefile = open("updated-scrape.csv","w",newline="")
writer = csv.writer(writefile)

for index, value in enumerate(price):
	if index == 0: 
		row_data = ["Time Instance","Price"]
		writer.writerow(row_data)
		continue
	writer.writerow([index,price[index]])
writefile.close()
pickle.dump(spread, open("spread.pk","wb"))
input("End")
os.system("python3 scrape_for_validation.py")




