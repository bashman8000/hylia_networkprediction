from os import listdir
import json
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

#e.g. link_name = "aofa_lond"
def plot_link(link_name):
    dfs[link_name].plot()
    plt.show()

def ts_to_date(ts):
    str_date = datetime.utcfromtimestamp(ts/1000).strftime('%Y-%m-%d %H:%M:%S')

    return datetime.strptime(str_date,'%Y-%m-%d %H:%M:%S')

files = ["link_data/" + f for f in listdir("link_data")]
#.DS_Store at index 0
files = files[1:]
dfs = {}

for i in range(len(files)):
    name = files[i]
    f = open(name)
    json_data = json.loads(f.read())
    df = pd.DataFrame(json_data["points"], dtype="float", columns=json_data["columns"])
    #make the index the timestamp
    df.set_index('time', inplace=True)
    #convert milisecond unix time to datetime
    df.index =  df.index.map(ts_to_date)
    #daily granularity
    #!!!DOUBLE CHECK THIS
    df = df.groupby(pd.Grouper(freq='D')).transform(np.cumsum).resample('D').median()
    #key value is the name of the link (e.g. from 'link_data/elpa_hous.txt' extract 'elpa_hous')
    dfs[name[10:len(name)-4]] = df

#links that have multiple interfaces
aofa_lond_summed = dfs.pop('aofa_lond-2-1-1').add(dfs.pop('aofa_lond-7-2-1'), fill_value=0)
dfs['aofa_lond'] = aofa_lond_summed
pnwg_denv_summed = dfs.pop('pnwg_denv-4-1-1').add(dfs.pop('pnwg_denv-10-1-4'), fill_value=0)
pnwg_denv_summed = pnwg_denv_summed.add(dfs.pop('pnwg_denv-10-1-6'))
dfs['pnwg_denv'] = pnwg_denv_summed

pacific_mtn = ['pnwg_denv', 'pnwg_bois', 'sacr_denv', 'sunn_lsvn', 'sunn_elpa']
mtn_ctrl = ['denv_kans', 'elpa_hous']
ctrl_east = ['star_bost', 'star_aofa', 'chic_wash', 'eqx-chi_eqx-ash', 'nash_wash', 'nash_atla']
trans_atl = ['bost_amst', 'newy_lond', 'aofa_lond', 'wash_cern-513']

zones = {'pacific_mtn': pacific_mtn, 'mtn_ctrl': mtn_ctrl, 'ctrl_east': ctrl_east, 'trans_atl': trans_atl}

#CODE TO NARROW DOWN TO 4 LINKS
# for zone in zones.keys():
#     links = zones[zone]
#     fig = plt.figure()
#     for link in links:
#         plt.plot(dfs[link].index,dfs[link]['in'], label=link+"_in")
#         plt.plot(dfs[link].index,dfs[link]['out'], label=link+"_out")
#     plt.legend()
#     plt.title(zone)
#     plt.show()

#AUTOCORRELATION FOR FINAL 4 LINKS
#ACF tells you how correlated points are with each other
#based on how many time steps they are separated by
#would expect it to fall towards 0 because it is harder to forecast into the future
sacr_denv_out = dfs['sacr_denv']['out']
denv_kans_in = dfs['denv_kans']['in']
star_aofa_in = dfs['star_aofa']['in']
aofa_lond_out = dfs['aofa_lond']['out']
final_4 = {'sacr_denv_out autocorrelation':sacr_denv_out, 'denv_kans_in autocorrelation': denv_kans_in, 'star_aofa_in autocorrelation': star_aofa_in, 'aofa_lond_out autocorrelation': aofa_lond_out}

for link in final_4.keys():
    data = final_4[link]
    plt.figure()
    autocorrelation_plot(data)
    plt.title(link)
    plt.show()





