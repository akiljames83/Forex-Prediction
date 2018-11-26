import pickle
txt = open("USDCAD.txt","r")

price_matrix = []

# 10k to 60k = 50k
price_first = []

# 2 Mil to 2.2 Mil = 200k
price_second = []

# 4 Mil to 4.2 mil = 200k
price_third = []

# 5.25 to 5.5 = 250k
price_fourth = []

# last ~200k pieces of data
price_validation = []

# first attempt, we will train on 10% of data with higher precedence given to more recent data
for index, line in enumerate(txt):
	if index == 0: continue
	data = line.split(",")[1:-1]
	price = float(data.pop(-1))

	# first array data
	if index >= 1e4 and index < 6e4: 
		price_first.append(price)

	# second array data
	elif index >= 2e6 and index < 2.2e6: 
		price_second.append(price)

	# third array data
	elif index >= 4e6 and index < 4.2e6: 
		price_third.append(price)

	# fourth array data
	elif index >= 5.25e6 and index < 5.5e6: 
		price_fourth.append(price)

	# fourth array data
	elif index >= 5.7e6: 
		price_validation.append(price)

	if index % 1000 == 0: print("Step %d completed." % index)

txt.close()

price_matrix.extend([price_first,price_second,price_third,price_fourth,price_validation])

pickle.dump(price_matrix, open("four_sub_categories_600k.pk", "wb"))