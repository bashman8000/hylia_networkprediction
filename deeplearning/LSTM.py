import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from lstm import *
from ts_algorithms import *

#get the df in shape
df = pd.read_csv("mawi-96hour.csv")
df = df[['Time', 'Traffic (GB)']]
df['Time'] = pd.to_datetime(df['Time'])
df['Traffic (GB)'] = pd.to_numeric(df['Traffic (GB)'])

scaler = MinMaxScaler(feature_range=(0,1))
traffic_scaler = MinMaxScaler(feature_range=(0,1))
scaler, traffic_scaler, X_train, Y_train, X_val, Y_val = train_test_wfeatures(df, scaler, traffic_scaler)

batch_size = 1

hw_preds = triple_exponential_smoothing(df['Traffic (GB)'][:72], 6, 0.7, 0.02, 0.9, 24)[72:]
arima_preds = arima(df['Traffic (GB)'][:72], 24)

#build and train the model
# nb_epoch = 20

# train_loss = []
# val_loss = []
# for i in range(nb_epoch):
#     #creating
#     model = Sequential()
#     model.add(LSTM(50, batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]), stateful=True))
#     model.add(Dense(1))
    
#     #building
#     model.compile(loss="mse",
#                       optimizer = 'adam',
#                     metrics = ['accuracy'])
    
#     #training
#     history = model.fit(X_train, Y_train, nb_epoch=1, batch_size=batch_size, shuffle=False, 
#                        validation_data = (X_val, Y_val), verbose = 0)
    
#     train_loss.append(history.history['loss'])
#     val_loss.append(history.history['val_loss'])
#     model.reset_states()

#does it work if i just create + build + train the model once?
#creating
model = Sequential()
model.add(LSTM(50, batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))

#building
model.compile(loss="mse", optimizer = 'adam', metrics = ['accuracy'])

#training
history = model.fit(X_train, Y_train, nb_epoch=1, batch_size=batch_size, shuffle=False, validation_data = (X_val, Y_val), verbose = 0)

model.reset_states()

#testing
total_truth, total_pred = train_val_predictions(model, X_train, Y_train, X_val, Y_val, 1)


# walk-forward validation on the test data
line_test_pred = np.reshape(total_pred, total_pred.shape[0])
line_test_real = np.reshape(total_truth, total_truth.shape[0])

#dates for plot
#dates = df_sacr_denv['Time'].apply(lambda x: datetime.fromtimestamp(x/1000.))
dates = df['Time']
# dates = df['Time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

fig1 = plt.figure(figsize=(20,10))
# plt.plot(dates, line_test_real, color='blue',label='Original', linewidth=1)
# plt.plot(dates, line_test_pred, color='red',label='Prediction', linewidth=1)
plt.plot(dates[72:], Y_val, color='blue',label='Original', linewidth=1)
plt.plot(dates[72:], total_pred[72:], color='red',label='lstm', linewidth=1)
plt.plot(dates[72:], hw_preds, color='green',label='hw', linewidth=1)
plt.plot(dates[72:], arima_preds, color='yellow',label='arima', linewidth=1)
# plt.axvline(x=dates[72], color = 'green')
plt.legend(loc='best')
plt.title('MAWI Predictions')
plt.show()
