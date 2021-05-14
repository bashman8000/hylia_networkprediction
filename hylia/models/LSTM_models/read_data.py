from os import listdir
import json
import simplejson
import pandas as pd
import matplotlib
from datetime import datetime
import numpy as np

#convert timestamp to datetime
def ts_to_date(ts):
    str_date = datetime.utcfromtimestamp(ts/1000).strftime('%Y-%m-%d %H:%M:%S')
    return datetime.strptime(str_date,'%Y-%m-%d %H:%M:%S')

#granularity can also be "daily"
#files: name of txt files
#return a dictionary mapping link name: link dataframe
def txts_to_dfs(files, granularity="hourly"):
    dfs = {}
    for i in range(len(files)):
        name = files[i]
        f = open(name)
        print f
        if(name=="link_data/.DS_Store"):
            print "ERROR"
        else:
            json_data = json.loads(f.read())
            #print json_data
            df = pd.DataFrame(json_data["points"], dtype="float", columns=json_data["columns"])
            #set index to time
            df.set_index('time', inplace=True)
            #convert milisecond unix time to datetime
            df.index =  df.index.map(ts_to_date)
            #daily granularity: use the median hour of the day (useful for initial plot)
            #!!!DOUBLE CHECK THIS
            if(granularity=="daily"):
                df = df.groupby(pd.Grouper(freq='D')).transform(np.cumsum).resample('D').median()
            #set link_name, e.g. from 'link_data/elpa_hous.txt' extract 'elpa_hous'
            link_name = name[10:len(name)-4]
            dfs[link_name] = df
    return dfs

#hardcoded with known link interfaces to combine
def combine_interfaces(dfs):
    #links that have multiple interfaces
    aofa_lond_summed = dfs.pop('aofa_lond-2-1-1').add(dfs.pop('aofa_lond-7-2-1'), fill_value=0)
    dfs['aofa_lond'] = aofa_lond_summed
    pnwg_denv_summed = dfs.pop('pnwg_denv-4-1-1').add(dfs.pop('pnwg_denv-10-1-4'), fill_value=0)
    pnwg_denv_summed = pnwg_denv_summed.add(dfs.pop('pnwg_denv-10-1-6'))
    dfs['pnwg_denv'] = pnwg_denv_summed
    return dfs
