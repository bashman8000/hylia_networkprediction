import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import pacf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from read_data import *
from ts_algorithms import *

files = ["link_data/" + f for f in listdir("link_data")]
#.DS_Store at index 0
files = files[1:]
#convert the texts to dataframes
#for the ones with multiple interfaces, sum over the interfaces
dfs = combine_interfaces(txts_to_dfs(files))

pacific_mtn = ['pnwg_denv', 'pnwg_bois', 'sacr_denv', 'sunn_lsvn', 'sunn_elpa']
mtn_ctrl = ['denv_kans', 'elpa_hous']
ctrl_east = ['star_bost', 'star_aofa', 'chic_wash', 'eqx-chi_eqx-ash', 'nash_wash', 'nash_atla']
trans_atl = ['bost_amst', 'newy_lond', 'aofa_lond', 'wash_cern-513']

zones = {'pacific_mtn': pacific_mtn, 'mtn_ctrl': mtn_ctrl, 'ctrl_east': ctrl_east, 'trans_atl': trans_atl}

#final dataframes for 4 links
#missing values filled in
sacr_denv_out = dfs['sacr_denv']['out']
sacr_denv_out = fill_missing(sacr_denv_out)
denv_kans_in = dfs['denv_kans']['in']
denv_kans_in = fill_missing(denv_kans_in)
star_aofa_in = dfs['star_aofa']['in']
star_aofa_in = fill_missing(star_aofa_in)
aofa_lond_out = dfs['aofa_lond']['out']
aofa_lond_out = fill_missing(aofa_lond_out)

#normalize
sacr_denv_out = normalize(sacr_denv_out)
denv_kans_in = normalize(denv_kans_in)
star_aofa_in = normalize(star_aofa_in)
aofa_lond_out = normalize(aofa_lond_out)

#training data
# sacr_denv_train = sacr_denv_out[:2136]
# denv_kans_train = denv_kans_in[:2136]
# star_aofa_train = star_aofa_in[:2136]
# aofa_lond_train = aofa_lond_out[:2136]
# sacr_denv_val = sacr_denv_out[2136:]
# denv_kans_val = denv_kans_in[:2136:]
# star_aofa_val = star_aofa_in[:2136:]
# aofa_lond_val = aofa_lond_out[:2136:]

#holt winters predictions
# sacr_denv_preds_hw = triple_exponential_smoothing(sacr_denv_train, 24, 0.7, 0.02, 0.9, 24)[2136:]
# denv_kans_preds_hw = triple_exponential_smoothing(denv_kans_train, 24, 0.7, 0.02, 0.9, 24)[2136:]
# star_aofa_preds_hw = triple_exponential_smoothing(star_aofa_train, 24, 0.7, 0.02, 0.9, 24)[2136:]
# aofa_lond_preds_hw = triple_exponential_smoothing(aofa_lond_train, 24, 0.7, 0.02, 0.9, 24)[2136:]

#arima predictions
#rules for identifying params: https://people.duke.edu/~rnau/arimrule.htm
#d=0 (no differencing) <-- series is stationary (no long term growth/decline rate) and small + patternless acfs
#q=0 (no MA) <-- no sharp cutoff in acf i.e. series is not overdifferenced
#p=1? <-- pcf is not showing anything...
# sacr_denv_preds_arima = arima(sacr_denv_train, 24)
# denv_kans_preds_arima = arima(denv_kans_train, 24)
# star_aofa_preds_arima = arima(star_aofa_train, 24)
# aofa_lond_preds_arima = arima(aofa_lond_train, 24)

#for mse table: hw
# for_mse_hw = [[sacr_denv_val, sacr_denv_preds_hw],
#             [denv_kans_val, denv_kans_preds_hw],
#             [star_aofa_val, star_aofa_preds_hw],
#             [aofa_lond_val, aofa_lond_preds_hw]]
# for_mse_arima = [[sacr_denv_val, sacr_denv_preds_arima],
#             [denv_kans_val, denv_kans_preds_arima],
#             [star_aofa_val, star_aofa_preds_arima],
#             [aofa_lond_val, aofa_lond_preds_arima]]
# mse_table_hw = avg_link_mses(for_mse_hw)
# mse_table_arima = avg_link_mses(for_mse_arima)

#for diff training sizes
# sacr_denv_mses = something(sacr_denv_out)
# denv_kans_mses = something(denv_kans_in)
# star_aofa_mses = something(star_aofa_in)
# aofa_lond_mses = something(aofa_lond_out)
# hw_mses = [sacr_denv_mses[0],
#            denv_kans_mses[0],
#            star_aofa_mses[0],
#            aofa_lond_mses[0]]
# arima_mses = [sacr_denv_mses[1],
#               denv_kans_mses[1],
#               star_aofa_mses[1],
#               aofa_lond_mses[1]] 
# hw_df = pd.DataFrame(hw_mses)
# arima_df = pd.DataFrame(arima_mses)
# hw_answer = dict(hw_df.mean())
# arima_answer = dict(arima_df.mean())

#---Justifications--- 
#Note: at the start of file, change txts_to_dfs(files) to txts_to_dfs(files,granularity="daily")

#---CODE TO NARROW DOWN TO 4 LINKS---
# for zone in zones.keys():
#     links = zones[zone]
#     fig = plt.figure()
#     for link in links:
#         plt.plot(dfs[link].index,dfs[link]['in'], label=link+"_in")
#         plt.plot(dfs[link].index,dfs[link]['out'], label=link+"_out")
#         plt.xlabel("Date", fontsize=16)
#         plt.ylabel("Traffic Data (GB)", fontsize=16)
#         plt.tick_params(labelsize=16)

#     plt.legend(prop={'size': 16})
#     print(zone)
#     #plt.title(zone)
#     plt.show()

#---CODE SHOWING ACF OR PACF FOR FINAL 4 LINKS---
# final_4 = {'sacr_denv_out autocorrelation':sacr_denv_out, 'denv_kans_in autocorrelation': denv_kans_in, 'star_aofa_in autocorrelation': star_aofa_in, 'aofa_lond_out autocorrelation': aofa_lond_out}
# for link in final_4.keys():
#     data = final_4[link]
#     plt.figure()
#     # autocorrelation_plot(data)
#     pacf(data)
#     # plt.tick_params(labelsize=16)
#     # plt.xlabel('Lag', fontsize=16)
#     # plt.ylabel('Autocorrelation', fontsize=16)
#     print(link)
#     # plt.title(link)
#     plt.show()




