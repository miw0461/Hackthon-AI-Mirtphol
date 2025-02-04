# hackthon-mirtphol 🚀
เป็นการออกแบบ AI วิเคราะห็ราคาน้ำตาลล่วงหน้า 1 ปี

## Code & Output ✨
  ### ver.1
  ใช้ Model Random Forest Tree ในการทำตัว AI
  - code : hackthon.py
  - import สิ่งที่ต้องใช้ 
    ```bash
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
  - เรียกใช้ load data.csv 
    ```bash
    # Load data
    data = pd.read_csv('data_Collection.csv')
  - Check for missing values 
    ```bash
    print(data.isnull().sum())
    
  - Remove commas and convert columns to numeric
    ```bash
    def clean_and_convert(column):
        column = column.str.replace(',', '')  # Remove commas
        column = pd.to_numeric(column, errors='coerce')  # Convert to numeric, set errors to NaN
        return column
    
  - Apply cleaning to all columns (or specify columns if necessary)
    ```bash
    for col in data.columns:
        if data[col].dtype == 'object':  # Only apply cleaning to object type columns
            data[col] = clean_and_convert(data[col])
    
  - Fill missing values
    ```bash
    data.ffill(inplace=True)  # This forward fills missing data
    
  - Select all columns except the target column for features
    ```bash
    selected_columns = [col for col in data.columns if col != 'sugar_price']
    selected_columns.append('sugar_price')
    
    df = data[selected_columns]
    
  - Split data into features (X) and target (y)
    ```bash
    X = df.drop(columns=['sugar_price'])
    y = df['sugar_price']
  - Split the data into training and testing sets
    ```bash
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
  - Create and train the model
    ```bash
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
  - Predict on test set
    ```bash
    y_pred = model.predict(X_test)
    
  - Evaluate model
    ```bash
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")
    
  - Predict future sugar prices for the next 365 days
     Start with the last available data
    ```bash
    current_data = X_test.iloc[-1].values.reshape(1, -1)
    current_data = pd.DataFrame(current_data, columns=X_train.columns)  # Add feature names
    
  - Predict future sugar prices
    ```bash
    future_prices = []
    ids = []  # Change variable name to 'ids'
    for i in range(365):
        next_price = model.predict(current_data)
        future_prices.append(next_price[0])
        ids.append(i + 1)  # Use numbers from 1 to 365
        
  - Update current_data with the predicted price and maintain feature names
    ```bash
        new_data = current_data.copy()
        new_data.iloc[0, :-1] = current_data.iloc[0, 1:]  # Shift data to the left
        new_data.iloc[0, -1] = next_price[0]  # Update with new prediction
        
        current_data = new_data
    
  - Create a DataFrame for future prices
    ```bash
    future_prices_df = pd.DataFrame({
        'ID': ids,  # Changed column name here
        'Prediction': future_prices
    })
    
  - Save the future prices DataFrame to a CSV file
    ```bash
    future_prices_df.to_csv('submission_v2.csv', index=False)
    
  - Display the future prices table
    ```bash
    print(future_prices_df)
    
  - Plot the predicted sugar prices
    ```bash
    plt.figure(figsize=(14, 7))
    plt.plot(future_prices_df['ID'], future_prices_df['Prediction'], marker='o', linestyle='-', color='b')  # Changed column name here
    plt.title('Predicted Sugar Prices for Next 365 Days')
    plt.xlabel('ID')  # Changed label here
    plt.ylabel('Prediction')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()
  - Output
  - ![image](https://github.com/user-attachments/assets/0046a6ae-83f6-4e9c-917b-e38de6914415)

## Code & Output ✨
  ###Ver.2
  - Code : test2.py
    ```bash
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      from statsmodels.tsa.arima.model import ARIMA
      from statsmodels.tsa.stattools import adfuller
      
  - Load data
    ```bash
      data = pd.read_csv('data_Collection.csv')
      
  - Data Preprocessing
    ```bash
      def clean_and_convert(column):
          column = column.str.replace(',', '')  # Remove commas
          column = pd.to_numeric(column, errors='coerce')  # Convert to numeric
          return column
      
      for col in data.columns:
          if data[col].dtype == 'object':
              data[col] = clean_and_convert(data[col])
      
      data.ffill(inplace=True)  # Fill missing values
      
  - Use only sugar price for ARIMA
    ```bash
      ts = data['sugar_price']
  - Check stationarity with ADF test
    ```bash
      adf_test = adfuller(ts)
      d = 0
      while adf_test[1] > 0.05 and d < 3:  # ลอง differencing ได้สูงสุด 3 รอบ
          ts = ts.diff().dropna()
          adf_test = adfuller(ts)
          d += 1
      
      
  - Differencing if needed
    ```bash
      d = 0 if adf_test[1] < 0.05 else 1
      
  - Train ARIMA model
    ```bash
      p, q = 5, 5  # Can be tuned using ACF/PACF plots
      model = ARIMA(ts, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
      model_fit = model.fit()
      
  - Predict for next 365 days
    ```bash
      future_steps = 365
      forecast = model_fit.forecast(steps=future_steps)
      
  - Create DataFrame
    ```bash
      future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1)[1:]
      predictions_df = pd.DataFrame({'Date': future_dates, 'Prediction': forecast})
      predictions_df.to_csv('arima_predictions.csv', index=False)
      
  - Plot results
    ```bash
      plt.figure(figsize=(14, 7))
      plt.plot(ts, label='Historical Data', color='blue')
      plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Forecast', color='red')
      plt.legend()
      plt.xlabel('Date')
      plt.ylabel('Sugar Price')
      plt.title('Predicted Sugar Prices for Next 365 Days')
      plt.grid(True)
      plt.show()
  - Output
  -![image](https://github.com/user-attachments/assets/3d499edb-828b-49de-95f5-8fb69f944fe1)
