#FFT ppy
from __future__ import print_function

from numpy.fft import fft,fftfreq
import json
import requests
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re
#%matplotlib inline
from scipy import signal
from statistics import mean


def data_to_df(link):
    filename = 'time-series-prediction/link_data/' + str(link) + '.txt'
    with open(filename) as f:
        data = json.load(f)
    df = pd.DataFrame(data['points'], columns=['Time', 'In', 'Out'])
    return df


    #loading data





df_sacr_denv = pd.read_csv('link_data_cleaned/sacr_denv_out.csv', header=None)
df_sacr_denv = df_sacr_denv.rename(columns = {0: "Time", 1:"Out"}) 

df_denv_kans = pd.read_csv('link_data_cleaned/denv_kans_in.csv', header=None)
df_denv_kans = df_denv_kans.rename(columns = {0: "Time", 1:"In"}) 

df_star_aofa = pd.read_csv('link_data_cleaned/star_aofa_in.csv', header=None)
df_star_aofa = df_star_aofa.rename(columns = {0: "Time", 1:"In"}) 

df_aofa_lond = pd.read_csv('link_data_cleaned/aofa_lond_out.csv', header=None)
df_aofa_lond = df_aofa_lond.rename(columns = {0: "Time", 1:"Out"})


df_aofa_lond.head()
print(df_aofa_lond)
x =0

fig1 = plt.figure(figsize=(20,10))

dates = df_sacr_denv['Time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
line_test_real = df_sacr_denv['Out']



#df = pd.read_csv("mawi-96hour.csv")
#df = df[['Time', 'Traffic (GB)']]
#df['Time'] = pd.to_datetime(df['Time'])
df_newGB= [47.68764941,
42.83059118,
40.72539717,
38.53067639,
35.7427651,
34.20400972,
32.42060629,
30.25938458,
25.70814667,
37.64572186,
36.94260301,
37.83281693,
39.81045442,
54.8156709,
55.10957692,
58.66307442,
60.66412956,
64.30165142,
60.67626959,
59.18075761,
55.79452423,
55.08510571,
58.96813613,
57.63638639,
56.48529677,
57.14166783,
50.19366259,
41.88931122,
40.49569576,
33.96741507,
29.68170325,
26.64273346,
28.00897663,
35.71928585,
38.78107129,
40.95161615,
43.24624163,
47.59857295,
49.79100291,
51.14168928,
54.42754855,
56.53997234,
56.75417982,
51.89412516,
53.17067596,
54.84579921,
51.63396779,
57.0315334,
49.0608246,
47.89428194,
37.682181,
35.90347762,
32.05108601,
28.96586631,
26.00435056,
30.76133575,
27.25953066,
31.34279779,
42.2549911,
45.94412998,
46.28680814,
43.34948927,
46.48666461,
49.74329861,
54.08093705,
58.53916972,
54.01875483,
50.38584292,
53.50541602,
47.25562739,
50.36676996,
63.57923394,
68.20750968,
62.51052929,
49.08102846,
43.37068494,
39.17866849,
38.44255047,
34.84553402,
33.51404468,
32.33744033,
39.10573701,
43.36486099,
58.2429928,
57.07129314,
51.60320919,
56.42639796,
50.44931734,
53.75922665,
55.27922302,
46.09663282,
44.785084,
54.13339608,
54.69659158,
58.99667236,
59.61858968]


#plt.plot(dates[x:], line_test_real, color='blue',label='Original', linewidth=1)
#plt.show()

#applying FFT
#print(df_newGB)
newy=line_test_real  #np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)
fft = np.fft.fft(newy)   #line_test_real)
fft_detrended = signal.detrend(fft)


for i in range(2):
    print("Value at index {}:\t{}".format(i, fft[i + 1]), "\nValue at index {}:\t{}".format(fft.size -1 - i, fft[-1 - i]))

N=2160 #number of sample points
T=1

t=np.linspace(0, 3, 2160) # 3 months with 2160 hours
T=0.001388888888889#t[1]-t[0]

print("T = ")
print(T)
# 1/T = frequency
#f = np.linspace(0, 2160/720,N)
f = np.linspace(0, 1/T,N)


plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")

loggingx=(f[:N // 2])
loggingy=np.abs(np.log10(fft_detrended[:N // 2] * 1 / N))
#plt.plot(loggingx, loggingy)  # 1 / N is a normalization factor

plt.bar(f[:N // 2], np.abs(fft_detrended)[:N // 2] * 1 / N, width=1.5)
plt.show()


period = 1/f[:N // 2] #loggingx
loggingy=np.abs(fft)[:N // 2] * 1 / N

plt.bar(period,loggingy, width =0.5);
plt.xlabel('Hours/Cycle')
plt.ylabel('Log(Power)')
plt.show()