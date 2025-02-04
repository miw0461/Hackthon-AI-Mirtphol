import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Load data
data = pd.read_csv('data_Collection.csv')

# Data Preprocessing
def clean_and_convert(column):
    column = column.str.replace(',', '')  # Remove commas
    column = pd.to_numeric(column, errors='coerce')  # Convert to numeric
    return column

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = clean_and_convert(data[col])

data.ffill(inplace=True)  # Fill missing values

# Use only sugar price for ARIMA
ts = data['sugar_price']


# Check stationarity with ADF test
adf_test = adfuller(ts)
d = 0
while adf_test[1] > 0.05 and d < 3:  # ลอง differencing ได้สูงสุด 3 รอบ
    ts = ts.diff().dropna()
    adf_test = adfuller(ts)
    d += 1


# Differencing if needed
d = 0 if adf_test[1] < 0.05 else 1

# Train ARIMA model
p, q = 5, 5  # Can be tuned using ACF/PACF plots
model = ARIMA(ts, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit()

# Predict for next 365 days
future_steps = 365
forecast = model_fit.forecast(steps=future_steps)

# Create DataFrame
future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1)[1:]
predictions_df = pd.DataFrame({'Date': future_dates, 'Prediction': forecast})
predictions_df.to_csv('arima_predictions.csv', index=False)

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(ts, label='Historical Data', color='blue')
plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Forecast', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Sugar Price')
plt.title('Predicted Sugar Prices for Next 365 Days')
plt.grid(True)
plt.show()
