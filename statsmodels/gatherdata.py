# gathering data about specific link and specific date

from tensorflow.python.client import device_lib

#print(device_lib.list_local_devices())
import requests
import json
import numpy as np
import pandas as pd
import datetime
import dateutil.parser
import csv



def run_query(query): # A simple function to use requests to make the API call. Returns json content from query.
	request = requests.get('https://my.es.net/graphql', json={'query': query})
	if request.status_code == 200:
		return request.json()
	else:
		raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))


def main():
	edge_id = 1894
	beginTime='2018-05-01T09:22:53.253Z'
	endTime='2018-08-31T18:22:53.253Z'

	query = '''
	{
		mapTopologyEdge(id: "%d") {
			name
		traffic(beginTime: "%s", endTime:"%s"){
			columns
			name
			points
			utc
			labels
			interface
			device
			sap
			tile
			}
		}
	}''' % (edge_id,beginTime,endTime)
	
	data = run_query(query)
	print(data)


	datastr=data["data"]["mapTopologyEdge"]["traffic"]
	e1,e2=datastr["labels"]
	cleaned_data=np.array(datastr["points"])
	print(cleaned_data)


	print("#####################")
	df=pd.DataFrame({'time': cleaned_data[:, 0], e1:cleaned_data[:, 1],e2:cleaned_data[:, 2]})
	print df

	export_csv=df.to_csv('pwngdenv.csv')



	
	


main()