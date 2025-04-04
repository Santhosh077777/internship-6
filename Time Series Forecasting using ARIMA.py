import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

date_rng = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
np.random.seed(42)
sales = np.random.poisson(lam=200, size=len(date_rng)) + np.sin(np.linspace(0, 50, len(date_rng))) * 20
df = pd.DataFrame({'Date': date_rng, 'Sales': sales})
df.to_csv('sales_data.csv', index=False)

df = pd.read_csv('/content/sales_time_series.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.asfreq('D')

plt.figure(figsize=(12, 5))
plt.plot(df['Sales'], label='Daily Sales')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

model = ARIMA(df['Sales'], order=(5, 1, 2))
model_fit = model.fit()

forecast = model_fit.forecast(steps=30)
forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Sales': forecast.values})

plt.figure(figsize=(12, 5))
plt.plot(df['Sales'], label='Historical Sales')
plt.plot(forecast_df['Date'], forecast_df['Predicted Sales'], color='red', label='Forecasted Sales')
plt.title('Sales Forecast for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid()
plt.show()

train = df.iloc[:-30]
test = df.iloc[-30:]

model_eval = ARIMA(train['Sales'], order=(5, 1, 2))
model_eval_fit = model_eval.fit()
preds = model_eval_fit.forecast(steps=30)

rmse = sqrt(mean_squared_error(test['Sales'], preds))
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
