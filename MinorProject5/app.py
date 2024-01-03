import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model
import streamlit as st
import yfinance as yfin
from datetime import date
import datetime

today = date.today()
yfin.pdr_override()

start = datetime.datetime(2012, 1, 1)

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = pdr.get_data_yahoo(user_input, start, today)

st.subheader("Date from 2012 - 2023")
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA &200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

data = df[['Close']].values

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare the data
X, y = [], []
for i in range(100, len(data_scaled)):
    X.append(data_scaled[i-100:i, 0])
    y.append(data_scaled[i, 0])
X, y = np.array(X), np.array(y)

# Reshape the data for LSTM input (samples, timesteps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# model = load_model('keras_LSTM_model.h5')
# last_100_days = data_scaled[-100:]
# last_100_days = np.reshape(last_100_days, (1, last_100_days.shape[0], 1))
#
# # Predict the next day's price
# predicted_price = model.predict(last_100_days)
# predicted_price = scaler.inverse_transform(predicted_price)
# print(predicted_price)


# Plotting the actual and predicted prices

# st.subheader(f"Prediction Price :$ {predicted_price[0][0]}")
# fig2=plt.figure(figsize=(12,6))
# plt.plot(data, label='Actual Prices')
# plt.plot(len(data) - 1, predicted_price, marker='o', color='red', label='Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.title('Stock Price Prediction for Next Day')
# plt.legend()
# st.pyplot(fig2)
st.subheader("Choose a Model")
selected_model = st.selectbox("Select a Model", ["LSTM", "CNN", "GRU", "Ensemble Model"])

# Fitting and predicting based on the selected model
if selected_model == "LSTM":
    model = load_model('keras_LSTM_model.h5')
    last_100_days = data_scaled[-100:]
    last_100_days = np.reshape(last_100_days, (1, last_100_days.shape[0], 1))
    predicted_price = model.predict(last_100_days)
    predicted_price = scaler.inverse_transform(predicted_price)

# ... (existing code)

elif selected_model == "CNN":
    model = load_model('keras_CNN_model.h5');
    last_100_days = data_scaled[-100:]
    last_100_days = np.reshape(last_100_days, (1, last_100_days.shape[0], 1))
    predicted_price = model.predict(last_100_days)
    predicted_price = scaler.inverse_transform(predicted_price)


elif selected_model == "GRU":
    model = load_model('keras_GRU_model.h5');
    last_100_days = data_scaled[-100:]
    last_100_days = np.reshape(last_100_days, (1, last_100_days.shape[0], 1))
    predicted_price = model.predict(last_100_days)
    predicted_price = scaler.inverse_transform(predicted_price)

elif selected_model == "Ensemble Model":
    model = load_model('keras_ENSEMBLe_model.h5');
    last_100_days = data_scaled[-100:]
    last_100_days = np.reshape(last_100_days, (1, last_100_days.shape[0], 1))
    predicted_price = model.predict(last_100_days)
    predicted_price = scaler.inverse_transform(predicted_price)


st.subheader(f"Prediction Price: $ {predicted_price[0][0]}")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual Prices')
plt.plot(len(data) - 1, predicted_price, marker='o', color='red', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction for Next Day')
plt.legend()
st.pyplot(fig2)
