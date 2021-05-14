import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, LSTM, Activation, Dropout, Bidirectional, TimeDistributed, RepeatVector, Input, GRU, Lambda
#from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
#from keras.utils import multi_gpu_model
import keras
from random import uniform
import json
# Fix AttributeError: 'module' object has no attribute 'control_flow_ops'
import tensorflow
from tensorflow.python.ops import control_flow_ops
tensorflow.control_flow_ops = control_flow_ops


def inverse_transform(pred, traffic_scaler):    
    return traffic_scaler.inverse_transform(pred)

def graph_results(model, X_test, Y_test, traffic_scaler, batch_size = 1):
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

#i added batch_size as an argument
def train_val_predictions(model, X_train, Y_train, X_val, Y_val, batch_size):
    X_train_pred = model.predict(X_train, batch_size)
    #X_train_pred_inv = inverse_transform(X_train_pred, scaler)

    X_val_pred = model.predict(X_val, batch_size)
    #X_val_pred_inv = inverse_transform(X_val_pred, scaler)

    y_train = np.float_(Y_train)
    #y_train_inv = inverse_transform(y_train, scaler)

    y_val = np.float_(Y_val)
    #y_val_inv = inverse_transform(y_val, traffic_scaler)

    total_truth = np.vstack((y_train, y_val))
    total_pred = np.vstack((X_train_pred, X_val_pred))
    
    return total_truth, total_pred

def train_test_wfeatures(df, scaler, traffic_scaler, print_shapes = True):
    """Returns training and test data from Pandas DataFrame of data

    Args:
        df (DataFrame): A Pandas DataFrame containing the data for one site to another site
                        the dataframe should contain time, features (ex. Day, Month, Weekend) 
                        and the response variable.
        pathway (str): The names of the two sites separated by a double dash (ex. "CHIC--STAR")
        split_proportion (float): Proportion (from 0 to 1) of data to be allocated to training data
        scaler (MinMaxScaler): scaler for features
        traffic_scaler (MinMaxScaler): scaler for response variable
        print_shapes (bool): True (default) to print shapes of the training and test data
                             False to turn off printing
        
        """
    
    #Split features from response variable
    X = df.iloc[:,1].as_matrix().reshape(-1,1) #drop time to get all features
    Y = df.iloc[:,1].shift(1).fillna(0).as_matrix().reshape(-1,1) #shift traffic values down 1 to create response variable

    #Normalize
    X = scaler.fit_transform(X)
    Y = traffic_scaler.fit_transform(Y)

    #reshape to [samples, features, timesteps]
    X = X.reshape(X.shape[0], 1, 1)
    Y = Y.reshape(Y.shape[0], 1)

    #Train-test split
    X_train = X[:72]
    Y_train = Y[:72]
    X_val = X[72:]
    Y_val = Y[72:]
    
    if print_shapes:
        print("X_train shape: ", X_train.shape)
        print("Y_train shape: ", Y_train.shape)
        print("X_val shape: ", X_val.shape)
        print("Y_val shape: ", Y_val.shape)
    
    return scaler, traffic_scaler, X_train, Y_train, X_val, Y_val