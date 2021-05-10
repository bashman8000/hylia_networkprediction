from tensorflow.python.client import device_lib
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
from tsfresh.feature_extraction import feature_calculators

tensorflow.control_flow_ops = control_flow_ops

def inverse_transform(pred, traffic_scaler):    
    return traffic_scaler.inverse_transform(pred)

def train_test_multistep(df, pathway, split_proportion, seq_len_x, seq_len_y, scaler, 
                         traffic_scaler, print_shapes = True):
    """Returns training and test data from Pandas DataFrame of data

    Args:
        df (DataFrame): A Pandas DataFrame containing the data for one site to another site
                        the dataframe should contain time, features (ex. Day, Month, Weekend) 
                        and the response variable.
        pathway (str): The names of the two sites separated by a double dash (ex. "CHIC--STAR")
        split_proportion (float): Proportion (from 0 to 1) of data to be allocated to training data
        seq_len_x (int): Number of time series observations to be included per training window
        seq_len_y (int): Number of time series observations to predict
        scaler (MinMaxScaler): scaler for features
        traffic_scaler (MinMaxScaler): scaler for response variable
        print_shapes (bool): True (default) to print shapes of the training and test data
                             False to turn off printing
        
        """
    
    #Split features from response variable
    X = df[[pathway]].as_matrix()
    X = scaler.fit_transform(X)
    
    result_X = []
    result_Y = []

    for index in range(0, X.shape[0] - (seq_len_x + seq_len_y + 1), seq_len_x + seq_len_y):
        result_X.append(X[index: index + seq_len_x]) #adding CHIC--STAR
        result_Y.append(X[index + seq_len_x: index + seq_len_x + seq_len_y])

    result_X = np.array(result_X)
    result_X = result_X.reshape(result_X.shape[0], result_X[0].shape[0])
    result_Y = np.array(result_Y)
    result_Y = result_Y.reshape(result_Y.shape[0], result_Y[0].shape[0])

    #Normalize
    #result_X = scaler.fit_transform(result_X)
    #result_Y = traffic_scaler.fit_transform(result_Y)

    #Train-test split
    row = int(round(split_proportion * result_X.shape[0]))
    X_train = result_X[:row]
    Y_train = result_Y[:row]
    X_test = result_X[row:]
    Y_test = result_Y[row:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    #X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1], 1))
    Y_test = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1], 1))
    #Y_train = Y_train.reshape(Y_train.shape[0], 1, Y_train.shape[1])
    #Y_test = Y_test.reshape(Y_test.shape[0], 1, Y_test.shape[1])

    if print_shapes:
        print("X_train shape: ", X_train.shape)
        print("Y_train shape: ", Y_train.shape)
        print("X_test shape: ", X_test.shape)
        print("Y_test shape: ", Y_test.shape)
    
    return X_train, Y_train, X_test, Y_test