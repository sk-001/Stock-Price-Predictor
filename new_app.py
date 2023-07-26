import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin
import streamlit as st
import datetime
from keras.models import load_model
import yfinance as yf
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import LSTM

import io

start = '2013-01-01'
end = '2023-07-14'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

yfin.pdr_override()

data = pdr.get_data_yahoo(user_input, start=start, end=end)

df=data.reset_index()['Close']

# Describing Data
st.subheader(f'Data from {start} to {end}')

st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df,label='Closing Price')
plt.xlabel('Day')
plt.ylabel('Closing Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 =  df.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df)
st.pyplot(fig)



st.subheader('Closing Price vs Time Chart with 100MA & 200MA')

ma100 =  df.rolling(100).mean()

ma200 =  df.rolling(200).mean()

fig = plt.figure(figsize=(12,6))

plt.plot(ma100, 'r')

plt.plot(ma200, 'g')

plt.plot(df, 'b')

st.pyplot(fig)

# scaling
# from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df).reshape(-1,1))

# splitting
# DATA PREPROCESSING
training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size

train_data=df1[0:training_size,:]
test_data=df1[training_size:len(df1),:1]

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)


model = load_model('1model.h5')

x_train = x_train[...,np.newaxis]
x_test = x_test[...,np.newaxis]


# hehehehhehe
y_predicted = model.predict(x_test)
# y_predicted = y_predicted.scaler.inverse_transform()
scaler = scaler.scale_

scale_factor = 1/scaler[0]

y_predicted = y_predicted * scale_factor

y_test = y_test * scale_factor

st.subheader('Predictions vs Original')

fig2 = plt.figure(figsize=(12,6))

plt.plot(y_test, 'b', label='Original Price')

plt.plot(y_predicted, 'r', label='Predicted Price')

plt.xlabel('Time')

plt.ylabel('Price')

plt.legend()

st.pyplot(fig2)


# nex 30 day
# train_predict=model.predict(x_train)
# test_predict=model.predict(x_test)
# train_predict=scaler.inverse_transform(train_predict)
# test_predict=scaler.inverse_transform(test_predict)

temp = len(test_data)
x_input=test_data[temp-100:].reshape(1,-1)
# x_input.shape
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=100
i=0
while(i<30):

    if(len(temp_input)>100):

        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))

        yhat = model.predict(x_input)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]

        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1)) 
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1


# print(lst_output)
st.subheader('forecasting for next 30 days')
temp = len(df1)
fig = plt.figure(figsize=(12,6))
df3=df1.tolist()
# print(type(df3))
scaler = scaler.scale_

scale_factor = 1/scaler[0]
df3=df3*scale_factor
df3=df3.tolist()
# print(type(df3))
# plt.plot(df3[10636:])
df3.extend(lst_output)
plt.plot(df3[temp-100:])
st.pyplot(fig)
