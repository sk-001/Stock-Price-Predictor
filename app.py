import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin
import streamlit as st
import datetime
from keras.models import load_model

start = '2013-01-01'
end = '2023-07-14'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

yfin.pdr_override()

df = pdr.get_data_yahoo(user_input, start=start, end=end)

# Describing Data
st.subheader(f'Data from {start} to {end}')

st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')

fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 =  df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)



st.subheader('Closing Price vs Time Chart with 100MA & 200MA')

ma100 =  df.Close.rolling(100).mean()

ma200 =  df.Close.rolling(200).mean()

fig = plt.figure(figsize=(12,6))

plt.plot(ma100, 'r')

plt.plot(ma200, 'g')

plt.plot(df.Close, 'b')

st.pyplot(fig)





# Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])

data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))



data_training_array = scaler.fit_transform(data_training)





# Load Model

model = load_model('1model.h5')



# Testing

past_100_days = data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)



x_test = []

y_test = []



for i in range(100,input_data.shape[0]):

    x_test.append(input_data[i-100: i])

    y_test.append(input_data[i, 0])



x_test, y_test = np.array(x_test), np.array(y_test)



y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]

y_predicted = y_predicted * scale_factor

y_test = y_test * scale_factor



# Predicted visualization

st.subheader('Predictions vs Original')

fig2 = plt.figure(figsize=(12,6))

plt.plot(y_test, 'b', label='Original Price')

plt.plot(y_predicted, 'r', label='Predicted Price')

plt.xlabel('Time')

plt.ylabel('Price')

plt.legend()

st.pyplot(fig2)


# predict for next 30 days
st.subheader('plot for next 30 days')
temp = len(data_testing)
print(data_testing.shape)
data_testing = data_testing.to_numpy()
data_testing.reshape(-1,1)
print(data_testing.shape)
x_input=data_testing[temp-100:].reshape(1,-1)
print(x_input.shape)
lst_output=[]
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
n_steps=100
i=0
while(i<30):

    if(len(temp_input)>100):

        x_input=np.array(temp_input[1:])
        # print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))

        yhat = model.predict(x_input)
        # print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]

        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1)) 
        yhat = model.predict(x_input, verbose=0)
        # print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        # print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

df3=df['Close'].to_numpy()

# df3=df3[0].tolist()
print(type(df3))
df3 = df3.reshape(-1,1)
print(df3.shape)
kk=len(df3)
df3=df3.tolist()
print(type(df3))


df3=scaler.inverse_transform(df3).tolist()
# plt.plot(df3[kk-100:])
df3.extend(lst_output)
fig5=plt.figure((12,6))
plt.plot(df3[kk-100:])
st.pyplot(fig5)
# print(lst_output)