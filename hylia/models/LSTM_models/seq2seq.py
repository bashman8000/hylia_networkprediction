#seq2seq
#stacked 2 LSTMs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils.vis_utils import plot_model
from keras.layers import Activation, Dropout
from keras.layers import Input, LSTM, Dense
from keras.models import Model

from keras.optimizers import Adam




from datetime import datetime
import re
import time
#%matplotlib inline
data_dim=24
batch_size=1
timesteps=8
#univariate feature
n_features=1
pred_steps=24


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)



def train_test_wfeatures(df, scaler, traffic_scaler, print_shapes = True):
	"""Returns training and test data from Pandas DataFrame of data"""
	print("3")
	#Split features from response variable
	X = df.iloc[:,1].values #dropping time
	Y = df.iloc[:,1].shift(1).fillna(0).values #shift traffic values  1 to create response variable

	#Normalize
	#X = scaler.fit_transform(X.reshape(-1,1))
	#Y = traffic_scaler.fit_transform(Y.reshape(-1,1))
	#print("values")
	#print(Y)
	c=np.array(Y)
	np.savetxt("real-Y.csv", c, delimiter=',')

	#reshape to [samples, features, timesteps]
	X = X.reshape(X.shape[0], 1, 1)
	Y = Y.reshape(Y.shape[0], 1)
	#Train-test split
	X_train = X[:2112]
	Y_train = Y[:2112]
	X_test = X[2136:]
	Y_test = Y[2136:]
	#X_test=X[2136:]
	#Y_test=Y[2136:]
	#print(X_val)
	"""
	X_train = X[:48]
	Y_train = Y[:48]
	X_val = X[48:72]
	Y_val = Y[48:72]
	X_test=X[72:]
	Y_test=Y[72:]
    """
	if print_shapes:
		print("X_train shape: ", X_train.shape)
		print("Y_train shape: ", Y_train.shape)
		#print("X_val shape: ", X_val.shape)
		#print("Y_val shape: ", Y_val.shape)
		print("X_test shape: ", X_test.shape)
		print("Y_test shape: ", Y_test.shape)

	return scaler, traffic_scaler, X_train, Y_train, X_test,Y_test



#################
def train_val_predictions(model, X_train, Y_train, X_val, Y_val,X_test,Y_test):
	print("2")
	start=time.perf_counter()
	print("values got")
	print(X_train.size, Y_train.size, X_val.size, Y_val.size,X_test.size,Y_test.size)
	
	X_train_pred = model.predict(X_train, batch_size) #X_train_pred_inv = inverse_transform(X_train_pred, scaler)
	X_val_pred = model.predict(X_val, batch_size) #X_val_pred_inv = inverse_transform(X_val_pred, scaler)
	X_test_pred = model.predict(X_test, batch_size)

	####
	print("VAL")
	for n in X_val_pred:
		print(n[0])

	print("XTEST")
	print(X_test_pred)

	onevalue=X_test_pred[0]
	newpredictions=[]
	newpredictions.append(onevalue)
	newpredictions2=X_test
	newpredictions2[0]=onevalue
	#newpredictions2=onevalue
	#newpredictions2[1]=onevalue

	#j=0
	for j in range(24):
		newarray=model.predict(newpredictions2, batch_size)
		onevalue=newarray[0]
		newpredictions2[0]=onevalue
		newpredictions.append(onevalue)
		#print("####")
		#print(onevalue)
	print("############")
	print("new 24 predictions")
	for j in newpredictions:
		print(j[0])



	end=time.perf_counter()
	

	y_train = np.float_(Y_train)#y_train_inv = inverse_transform(y_train, scaler)
	y_val = np.float_(Y_val)#y_val_inv = inverse_transform(y_val, traffic_scaler)
	y_test = np.float_(Y_test)


	total_truth = np.vstack((y_train, y_val))	
	total_pred = np.vstack((X_train_pred, X_val_pred))
	total_pred_test = np.vstack((X_train_pred, X_test_pred))
	print("Time per prediction:")


	end=time.perf_counter()
	
	y_train = np.float_(Y_train)#y_train_inv = inverse_transform(y_train, scaler)
	y_val = np.float_(Y_val)#y_val_inv = inverse_transform(y_val, traffic_scaler)
	y_test = np.float_(Y_test)


	total_truth = np.vstack((y_train, y_val))	
	total_pred = np.vstack((X_train_pred, X_val_pred))
	total_pred_test = np.vstack((X_train_pred, X_test_pred))

	print("Time per prediction:")

	diff= end-start
	print(diff) 


	print(total_truth.size,total_pred.size, total_pred_test.size)

	return total_truth, total_pred, total_pred_test

# load the dataset
#dataframe = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')
#dataset = dataframe.values
#dataset = dataset.astype('float32')

def inverse_transform(pred, traffic_scaler):    
    return traffic_scaler.inverse_transform(pred)

def graph_results(model,X_val, Y_val, X_test, Y_test, traffic_scaler, batch_size = 1):
    # walk-forward validation on the test data
    pred_x_test = model.predict(X_test, batch_size)
    pred_test = inverse_transform(pred_x_test, traffic_scaler)

    y_test = np.float_(Y_test)
    y_test_inv = inverse_transform(y_test, traffic_scaler)

    line_test_pred = np.reshape(pred_test, pred_test.shape[0])
    line_test_real = np.reshape(y_test_inv, y_test_inv.shape[0])
    plt.figure(figsize=(20,10))
    plt.plot(line_test_real, color='blue',label='Original', linewidth=1)
    plt.plot(line_test_pred, color='red',label='Prediction', linewidth=1)
    plt.legend(loc='best')
    plt.title('Test - Comparison')
    plt.show()


def transform_series_encode(series_array):
    
    series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0
    series_mean = series_array.mean(axis=1).reshape(-1,1) 
    series_array = series_array - series_mean
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))
    return series_array, series_mean

def transform_series_decode(series_array, encode_series_mean):
    
    series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0
    series_array = series_array - encode_series_mean
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))
    
    return series_array


def decode_sequence(encoder_model, decoder_model, input_seq, pred_steps):
    
    # Encode the input as state vectors.
    print("input_seq")
    print(input_seq)

    states_value = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 1))
    
    # Populate the first target sequence with end of encoding series pageviews
    target_seq[0, 0, 0] = input_seq[0, -1, 0]

    # Sampling loop for a batch of sequences - we will fill decoded_seq with predictions
    # (to simplify, here we assume a batch of size 1).

    decoded_seq = np.zeros((1,pred_steps,1))
    
    for i in range(pred_steps):
        
        output, h, c = decoder_model.predict([target_seq] + states_value)
        
        decoded_seq[0,i,0] = output[0,0,0]

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 1))
        target_seq[0, 0, 0] = output[0,0,0]

        # Update states
        states_value = [h, c]

    return decoded_seq


def predict_and_plot(encoder_model, decoder_model, encoder_input_data, decoder_target_data, sample_ind, enc_tail_len=50):

	steps_predict=24
	sample_ind=24
	encode_series = encoder_input_data[sample_ind:sample_ind+1,:,:] 
	print("encode_series")
	print(encode_series)
	print(encode_series.shape)
	pred_series = decode_sequence(encoder_model, decoder_model, encode_series, steps_predict)
    
	encode_series = encode_series.reshape(-1,1)
	pred_series = pred_series.reshape(-1,1)   
	target_series = decoder_target_data[sample_ind:,:1].reshape(-1,1) 
	print("pred_series")
	print(pred_series)
	print("target_series")
	print(target_series)
    
	encode_series_tail = np.concatenate([encode_series[-enc_tail_len:],target_series[:1]])
	x_encode = encode_series_tail.shape[0]
    
	plt.figure(figsize=(10,6))   
    
	plt.plot(range(1,x_encode+1),encode_series_tail)
	plt.plot(range(x_encode,x_encode+pred_steps),target_series,color='orange')
	plt.plot(range(x_encode,x_encode+pred_steps),pred_series,color='teal',linestyle='--')
    
	plt.title('Encoder Series Tail of Length %d, Target Series, and Predictions' % enc_tail_len)
	plt.legend(['Encoding Series','Target Series','Predictions'])



def main():
	df_aofa_lond = pd.read_csv('../trans_atl/lond_newy_out.csv', header=None)
	#df_aofa_lond = df_aofa_lond.rename(columns = {0: "Time", 1:"fifteeen", 2:"thirty", 
	#	3:"fortyfive", 4:"total", 5:"Out"})
	#df_aofa_lond['Time'] = pd.to_datetime(df_aofa_lond['Time'])
	df_aofa_lond = df_aofa_lond.rename(columns = {0: "Time", 1:"Out"})
	dates = df_aofa_lond['Time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

	"""
	df_aofa_lond = pd.read_csv('../trans_atl/mawi-96hour.csv', header=None)
	df_aofa_lond = df_aofa_lond.rename(columns = {0: "Time", 1:"fifteeen", 2:"thirty", 
		3:"fortyfive", 4:"total", 5:"Out"})
	"""


	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	traffic_scaler = MinMaxScaler(feature_range=(0,1))
	#create train, val, test data
	print("4")
	scaler, traffic_scaler, X_train_linkdata, Y_train_linkdata, X_test_linkdata,Y_test_linkdata = train_test_wfeatures(df_aofa_lond[['Time', 'Out']], scaler, traffic_scaler)
	print("1")

	time_callback=TimeHistory()
	latent_dim=48
	dropout=0.2
	
	# create and fit the LSTM network - 1 layer
	# Define an input series and encode it with an LSTM. 
	encoder_inputs = Input(shape=(None, 1)) 
	encoder = LSTM(latent_dim, dropout=dropout, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)

	# We discard `encoder_outputs` and only keep the final states. These represent the "context"
	# vector that we use as the basis for decoding.
	encoder_states = [state_h, state_c]
	# Set up the decoder, using `encoder_states` as initial state.
	# This is where teacher forcing inputs are fed in.
	decoder_inputs = Input(shape=(None, 1)) 

	#We set up our decoder using `encoder_states` as initial state.  
	# We return full output sequences and return internal states as well. 
	# We don't use the return states in the training model, but we will use them in inference.
	decoder_lstm = LSTM(latent_dim, dropout=dropout, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)

	decoder_dense = Dense(1) # 1 continuous output at each timestep
	decoder_outputs = decoder_dense(decoder_outputs)

	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	model.summary()




	#first_n_samples=2112
	batch_size=24
	epochs=3


 	#sample of series from train_enc_start to train_enc_end  
	encoder_input_data = X_train_linkdata
	encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)

	decoder_target_data = Y_train_linkdata

	decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)

	# lagged target series for teacher forcing
	decoder_input_data = np.zeros(decoder_target_data.shape)
	decoder_input_data[:,1:,0] = decoder_target_data[:,:-1,0]
	decoder_input_data[:,0,0] = encoder_input_data[:,-1,0]

	model.compile(Adam(), loss='mean_absolute_error')
	history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_split=0.2)


	#build mode;
	#model = Sequential()
	#return 24 vector points 
	#model.add(LSTM(24, activation='tanh', return_sequences=True, batch_input_shape=(batch_size, X_train_linkdata.shape[1],X_train_linkdata.shape[2]), stateful=True))
	#model.add(Dropout(0.5))

	#model.add(LSTM(24, activation='tanh',return_sequences=True))
	#model.add(Dropout(0.5))

	#model.add(Flatten())

	#model.add(Dense(1)) #fitting to one class output predicting t+1
	#model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
	
	print(model.summary())
	#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


	#hist=model.fit(X_train_linkdata, Y_train_linkdata, epochs=10, batch_size=batch_size, shuffle=False, validation_data = (X_val_linkdata, Y_val_linkdata), verbose = 0, callbacks=[time_callback])
	
	#model.reset_states()


	#print(hist.history)#outputs val_loss, val_acc, acciuracy
	#print(hist.epoch)#number of epochs 10
	#print("epoch times:")
	#print(time_callback.times) #time per epoch

	sumt=0
	#for t in time_callback.times:
	#	sumt=sumt+t
	#sumt=sumt/10
	print("Avergae time per epoch:")
	print(sumt)

	plt.plot(history.history['loss'])
	print("loss")
	print(history.history['loss'])
	plt.plot(history.history['val_loss'])
	print(history.history['val_loss'])

	plt.xlabel('Epoch')
	plt.ylabel('Mean Absolute Error Loss')
	plt.title('Loss Over Time')
	plt.legend(['Train','Valid'])
	plt.show()

	# make predictions

	# from our previous model - mapping encoder sequence to state vectors
	encoder_model = Model(encoder_inputs, encoder_states)

	# A modified version of the decoding stage that takes in predicted target inputs
	# and encoded state vectors, returning predicted target outputs and decoder state vectors.
	# We need to hang onto these state vectors to run the next step of the inference loop.
	decoder_state_input_h = Input(shape=(latent_dim,))
	decoder_state_input_c = Input(shape=(latent_dim,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]

	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs] + decoder_states)



	encoder_input_data =  X_test_linkdata
	print("test input")
	print(X_test_linkdata)
	encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)


	decoder_target_data =  Y_test_linkdata

	decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)
	print("test output")
	print(decoder_target_data)
	predict_and_plot(encoder_model,decoder_model, encoder_input_data, decoder_target_data, 24)
	print("predict")



	graph_results(model,X_val_linkdata, Y_val_linkdata, X_test_linkdata, Y_test_linkdata, traffic_scaler)

	total_truth_linkdata, total_pred_linkdata, total_pred_test_linkdata = train_val_predictions(model, X_train_linkdata, Y_train_linkdata,X_val_linkdata, Y_val_linkdata, X_test_linkdata, Y_test_linkdata)
	x=1
	print("6")

	# walk-forward validation on the test data
	line_test_pred = np.reshape(total_pred_linkdata[x:], total_pred_linkdata[x:].shape[0])
	line_test_real = np.reshape(total_truth_linkdata[x:], total_truth_linkdata[x:].shape[0])
	line_test_real_withtest = np.reshape(total_pred_test_linkdata[x:], total_pred_test_linkdata[x:].shape[0])

	print(line_test_pred.size,line_test_real.size,line_test_real_withtest.size)
	print("7")
	a=np.array([line_test_pred,line_test_real,line_test_real_withtest]).T
	np.savetxt("foo.csv", a, delimiter=',')


	actual_predictions=line_test_real_withtest[0:]
	print(actual_predictions)
	#print("correspondingvalues")
	#data_v = traffic_scaler.inverse_transform(actual_predictions.reshape(-1,1))
	#print(data_v)
	b=np.array(actual_predictions)
	np.savetxt("real2.csv", b, delimiter=',')
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
