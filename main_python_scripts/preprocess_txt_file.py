'''
Creating the CSV processed data file from the USDCAD.txt document to be then processed with pandas.

'''
import csv
from pandas import read_csv

txt = open("USDCAD.txt","r")
writefile = open("USDCAD.csv","w",newline="")
writer = csv.writer(writefile)

for index, line in enumerate(txt):
	if index == 0: 
		row_data = ["Time Instance","Price"]
		writer.writerow(row_data)
		continue
	data = line.split(",")[1:-1]
	#print(data)
	date = data[0]
	time = data[1]
	price = data.pop(-1)

	date = date[2:]
	date_formatted = "-".join([date[i:i+2] for i in range(0,len(date),2)])
	time_formatted = ":".join([time[i:i+2] for i in range(0,len(time),2)])
	#print(date_formatted,"\n",time_formatted)
	row_data = ["|".join([date_formatted,time_formatted]),price]
	print(row_data)
	#print(instance,"\t",price)
	# YYMMDDHHmmSS

	# Write the data to the CSV file
	writer.writerow(row_data)
	if index > 500: break


writefile.close()
txt.close()