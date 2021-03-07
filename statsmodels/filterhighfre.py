from __future__ import print_function

import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt


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



df_sacr_denv = pd.read_csv('link_data_cleaned/sacr_denv_out.csv', header=None)
df_sacr_denv = df_sacr_denv.rename(columns = {0: "Time", 1:"Out"}) 

df_denv_kans = pd.read_csv('link_data_cleaned/denv_kans_in.csv', header=None)
df_denv_kans = df_denv_kans.rename(columns = {0: "Time", 1:"In"}) 

df_star_aofa = pd.read_csv('link_data_cleaned/star_aofa_in.csv', header=None)
df_star_aofa = df_star_aofa.rename(columns = {0: "Time", 1:"In"}) 

df_aofa_lond = pd.read_csv('trans_atl/amst_bost_out.csv', header=None)
df_aofa_lond = df_aofa_lond.rename(columns = {0: "Time", 1:"Out"})


df_aofa_lond.head()
#print(df_aofa_lond)
x =0

#fig1 = plt.figure(figsize=(20,10))

dates = df_aofa_lond['Time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
line_test_real = df_aofa_lond['Out']



newy=line_test_real  #np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)


# Seed the random number generator


time_step = 0.001388888888889

time_vec = np.arange(0, 3, time_step)
sig = newy

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
plt.xlabel('Time [Month]')
plt.ylabel('Traffic (GB)')
plt.show()

print("1")
# The FFT of the signal
sig_fft = fftpack.fft(sig)

# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)

# The corresponding frequencies
sample_freq = fftpack.fftfreq(sig.size, d=time_step)

# Plot the FFT power
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Sample Frequencies identified [Hz]')
plt.ylabel('Power')

# Find the peak frequency: we can focus on only the positive frequencies
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]

# Check that it does indeed correspond to the frequency that we generate
# the signal with
#np.allclose(peak_freq, 1./period)

# An inner plot to show the peak frequency
axes = plt.axes([0.55, 0.3, 0.3, 0.5])
plt.title('Peak frequencies')
plt.plot(freqs[:8], power[:8])
plt.setp(axes, yticks=[])

# scipy.signal.find_peaks_cwt can also be used for more advanced
# peak detection


high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig, linewidth=3, label='Filtered signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.legend(loc='best')
plt.show()




# Number of samplepoints
N = 2160
# sample spacing
T = 0.001388888888889
x = np.linspace(0.0, N*T, N)
y = sig
yf = fftpack.fft(y)
#xf = np.linspace(0, 3, time_step)
h=2/N * np.abs(yf[0:N])
print(h)

plt.subplot(2, 1, 1)
plt.plot(x, 2/N * np.abs(yf[0:N]))
plt.subplot(2, 1, 2)
plt.plot(x[1:], 2.0/N * np.abs(yf[0:N])[1:])
plt.show()



