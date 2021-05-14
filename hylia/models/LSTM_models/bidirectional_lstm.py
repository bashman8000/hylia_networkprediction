#stacked 2 LSTMs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM

from keras.layers import Activation, Dropout, Bidirectional, TimeDistributed, RepeatVector, Input, GRU, Lambda
#
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from datetime import datetime
import re
#%matplotlib inline
data_dim=24
batch_size=1
timesteps=8
#univariate feature
n_features=1


def train_test_wfeatures(df, scaler, traffic_scaler, print_shapes = True):
	"""Returns training and test data from Pandas DataFrame of data"""
	print("3")
	#Split features from response variable
	X = df.iloc[:,1].values #dropping time
	Y = df.iloc[:,1].shift(1).fillna(0).values #shift traffic values  1 to create response variable

	#Normalize
	X = scaler.fit_transform(X.reshape(-1,1))
	Y = traffic_scaler.fit_transform(Y.reshape(-1,1))
	#reshape to [samples, features, timesteps]
	X = X.reshape(X.shape[0], 1, 1)
	Y = Y.reshape(Y.shape[0], 1)
	#Train-test split
	X_train = X[:2112]
	Y_train = Y[:2112]
	X_val = X[2112:2136]
	Y_val = Y[2112:2136]
	X_test=X[2136:]
	Y_test=Y[2136:]

	if print_shapes:
		print("X_train shape: ", X_train.shape)
		print("Y_train shape: ", Y_train.shape)
		print("X_val shape: ", X_val.shape)
		print("Y_val shape: ", Y_val.shape)
		print("X_test shape: ", X_test.shape)
		print("Y_test shape: ", Y_test.shape)

	return scaler, traffic_scaler, X_train, Y_train, X_val, Y_val,X_test,Y_test



#################
def train_val_predictions(model, X_train, Y_train, X_val, Y_val,X_test,Y_test):
	print("2")
	X_train_pred = model.predict(X_train, batch_size) #X_train_pred_inv = inverse_transform(X_train_pred, scaler)
	X_val_pred = model.predict(X_val, batch_size) #X_val_pred_inv = inverse_transform(X_val_pred, scaler)
	X_test_pred = model.predict(X_test, batch_size)
	y_train = np.float_(Y_train)#y_train_inv = inverse_transform(y_train, scaler)
	y_val = np.float_(Y_val)#y_val_inv = inverse_transform(y_val, traffic_scaler)
	y_test = np.float_(Y_test)


	total_truth = np.vstack((y_train, y_val))	
	total_pred = np.vstack((X_train_pred, X_val_pred))
	total_pred_test = np.vstack((X_train_pred, X_test_pred))

	print(total_truth.size,total_pred.size, total_pred_test.size)

	return total_truth, total_pred, total_pred_test

# load the dataset
#dataframe = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')
#dataset = dataframe.values
#dataset = dataset.astype('float32')

def main():
	df_aofa_lond = pd.read_csv('../link_data_cleaned/aofa_lond_out.csv', header=None)
	df_aofa_lond = df_aofa_lond.rename(columns = {0: "Time", 1:"Out"})
	dates = df_aofa_lond['Time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
	actual_data = df_aofa_lond['Out']

	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	traffic_scaler = MinMaxScaler(feature_range=(0,1))
	#create train, val, test data
	print("4")
	scaler, traffic_scaler, X_train_linkdata, Y_train_linkdata, X_val_linkdata, Y_val_linkdata,X_test_linkdata,Y_test_linkdata = train_test_wfeatures(df_aofa_lond[['Time', 'Out']], scaler, traffic_scaler)
	print("1")
	
	# create and fit the LSTM network - 1 layer
	#build mode;
	model = Sequential()
	#return 24 vector points 
	model.add(Bidirectional(LSTM(24), batch_input_shape=(batch_size, X_train_linkdata.shape[1],X_train_linkdata.shape[2])))
	model.add(Flatten())
	model.add(Dense(1)) #fitting to one class output predicting t+1
	model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
	
	print(model.summary())

	model.fit(X_train_linkdata, Y_train_linkdata, epochs=1, batch_size=batch_size, shuffle=False, validation_data = (X_val_linkdata, Y_val_linkdata), verbose = 0)


	# make predictions
	total_truth_linkdata, total_pred_linkdata, total_pred_test_linkdata = train_val_predictions(model, X_train_linkdata, Y_train_linkdata,X_val_linkdata, Y_val_linkdata, X_test_linkdata, Y_test_linkdata)
	x=1900
	print("6")

	# walk-forward validation on the test data
	line_test_pred = np.reshape(total_pred_linkdata[x:], total_pred_linkdata[x:].shape[0])
	line_test_real = np.reshape(total_truth_linkdata[x:], total_truth_linkdata[x:].shape[0])
	line_test_real_withtest = np.reshape(total_pred_test_linkdata[x:], total_pred_test_linkdata[x:].shape[0])

	print(line_test_pred.size,line_test_real.size,line_test_real_withtest.size)
	print("7")
	a=np.array([line_test_pred,line_test_real,line_test_real_withtest]).T
	np.savetxt("foo.csv", a, delimiter=',')
	#dates for plot
	#dates = df_sacr_denv['Time'].apply(lambda x: datetime.fromtimestamp(x/1000.))
	dates = df_aofa_lond['Time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

	moreX = df_aofa_lond[['Time', 'Out']].iloc[:,1].values
	moreX_add = moreX[2136:]
	line_test_real=np.append(line_test_real,moreX_add)

	fig1 = plt.figure(figsize=(20,10))
	plt.plot(dates[x:], line_test_real, color='blue',label='Original', linewidth=1)
	print(line_test_real.size)
	print("5")

	line_test_real_withtest=np.append(line_test_real_withtest,moreX_add)
	plt.plot(dates[x:], line_test_real_withtest, color='red',label='Prediction', linewidth=1)
	plt.axvline(x=dates[2136], color = 'green')
	plt.legend(loc='best')
	plt.title('Predictions')
	plt.show()

main()
