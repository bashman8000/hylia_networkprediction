#S2STP model

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, LSTM, Activation, Dropout, Bidirectional, TimeDistributed, RepeatVector
from keras.layers import Input, GRU, Lambda, PReLU, concatenate, multiply, Activation
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute
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

import os
import json
import simplejson

import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from read_data import *
from ts_algorithms import *
from numpy import split
from numpy import array

def extract_date_features(linkway):
    #Get only one pathway (CHIC--STAR)
   # single_pathway = df[['Time', pathway]]

    #Convert times to datetime objects
    #times = single_pathway['Time']

    #Is date weekend?
    weekends = times.apply(lambda x: x.weekday() >= 5)*1
    single_pathway['Weekend'] = weekends

    #Add day of the week
    days = times.apply(lambda x: x.weekday())
    single_pathway['Day'] = days

    #Add hours, minutes
    hours = times.apply(lambda x: x.hour)
    minutes = times.apply(lambda x: x.minute + x.second/60.)
    single_pathway['Hour'] = hours
    single_pathway['Minute'] = minutes

    return single_pathway

def read_links():
    print "in read_link"
    files = ["link_data/" + f for f in listdir("link_data")]
    #.DS_Store at index 0
    files = files[1:]
    #convert the the texts to dataframes
    #for the ones with multiple interfaces, sum over the interfaces

    #print files
    dfs = combine_interfaces(txts_to_dfs(files))
    #print dfs
    pacific_mtn = ['pnwg_denv', 'pnwg_bois', 'sacr_denv', 'sunn_lsvn', 'sunn_elpa']
    mtn_ctrl = ['denv_kans', 'elpa_hous']
    ctrl_east = ['star_bost', 'star_aofa', 'chic_wash', 'eqx-chi_eqx-ash', 'nash_wash', 'nash_atla']
    trans_atl = ['bost_amst', 'newy_lond', 'aofa_lond', 'wash_cern-513']

    zones = {'pacific_mtn': pacific_mtn, 'mtn_ctrl': mtn_ctrl, 'ctrl_east': ctrl_east, 'trans_atl': trans_atl}

    #this is the relevant data: the dfs corresponding to the 4 links that we are using
    sacr_denv_out = dfs['sacr_denv']['out']
    denv_kans_in = dfs['denv_kans']['in']
    star_aofa_in = dfs['star_aofa']['in']
    aofa_lond_out = dfs['aofa_lond']['out']

      # print sacr_denv_out
    return sacr_denv_out


def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each hour
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores

#90 days gives about 12 weeks

# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    train, test = data[1:-328], data[-328:-6]
    # restructure into windows of weekly data
    train = array(split(train, len(train)))
    test = array(split(test, len(test)))
    return train, test



def main():
	#load the link data
    chosen_link=read_links()

    print "Link chosen"
    chosen_link.columns=['time','traffic']

    for v in chosen_link:
        print(v)

    for t in chosen_link['time']:
        print(t)
    #print chosen_link
    #print chosen_link.columns

    #plot data
    plt=chosen_link.plot(x='time', y='data',style='o')
    

    print(chosen_link.shape)
    train,test=split_dataset(chosen_link.values)
    #validate train data
    print(train.shape)
    print(train[0, 0], train[-1, -1])
    # validate test
    print(test.shape)
    print(test[0, 0], test[-1, -1])


main()
