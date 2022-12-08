import requests
import numpy as np
from sklearn.linear_model import LinearRegression
import csv
import datetime
import pandas as pd

# Set the time interval to 5 minutes
time_interval = 5

# Get the BTC price data for the past day from the CryptoCompare API
data = requests.get('https://min-api.cryptocompare.com/data/histominute?fsym=BTC&tsym=USD&limit=1440&aggregate=' + str(time_interval)).json()

# Extract the prices from the data
prices = [x['close'] for x in data['Data']]

# Calculate the relative strength index (RSI)
# Formula: RSI = 100 - (100 / (1 + RS))
# where RS = average_gain / average_loss

# Calculate the average gain and loss for the past 14 periods
periods = 14

# Calculate the price change for each period
price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

# Calculate the average gain and loss for the past 14 periods
average_gain = np.mean([x for x in price_changes if x > 0])
average_loss = abs(np.mean([x for x in price_changes if x < 0]))

# Calculate the relative strength (RS)
RS = average_gain / average_loss

# Calculate the relative strength index (RSI)
RSI = 100 - (100 / (1 + RS))

# Use linear regression to predict the BTC price after 1 hour
X = [[i] for i in range(len(prices))]
y = prices

# Train the model
model = LinearRegression()
model.fit(X, y)

# Use the model to predict the BTC price after 1 hour
prediction = model.predict([[len(prices) + (60 / time_interval)]])

print('Predicted BTC price after 1 hour:', prediction[0])

# Open the CSV file for writing
with open('predictions.csv', 'w', newline='') as csvfile:
    
    # Create a CSV writer
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(['timestamp', 'prediction'])

    # Write the prediction
    writer.writerow([datetime.datetime.now(), prediction[0]])