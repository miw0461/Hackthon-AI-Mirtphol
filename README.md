# hackthon-mirtphol 🚀
เป็นการออกแบบ AI วิเคราะห็ราคาน้ำตาลล่วงหน้า 1 ปี

## Code & Output ✨
### ver.1
- code
  ```bash
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import mean_squared_error
  import matplotlib.pyplot as plt
  
  # Load data
  data = pd.read_csv('data_Collection.csv')

  # Check for missing values
  print(data.isnull().sum())
  
  # Remove commas and convert columns to numeric
  def clean_and_convert(column):
      column = column.str.replace(',', '')  # Remove commas
      column = pd.to_numeric(column, errors='coerce')  # Convert to numeric, set errors to NaN
      return column
  
  # Apply cleaning to all columns (or specify columns if necessary)
  for col in data.columns:
      if data[col].dtype == 'object':  # Only apply cleaning to object type columns
          data[col] = clean_and_convert(data[col])
  
  # Fill missing values
  data.ffill(inplace=True)  # This forward fills missing data
  
  # Select all columns except the target column for features
  selected_columns = [col for col in data.columns if col != 'sugar_price']
  selected_columns.append('sugar_price')
  
  df = data[selected_columns]
  
  # Split data into features (X) and target (y)
  X = df.drop(columns=['sugar_price'])
  y = df['sugar_price']
  
  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
  
  # Create and train the model
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)
  
  # Predict on test set
  y_pred = model.predict(X_test)
  
  # Evaluate model
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  print(f"RMSE: {rmse}")
  
  # Predict future sugar prices for the next 365 days
  # Start with the last available data
  current_data = X_test.iloc[-1].values.reshape(1, -1)
  current_data = pd.DataFrame(current_data, columns=X_train.columns)  # Add feature names
  
  # Predict future sugar prices
  future_prices = []
  ids = []  # Change variable name to 'ids'
  for i in range(365):
      next_price = model.predict(current_data)
      future_prices.append(next_price[0])
      ids.append(i + 1)  # Use numbers from 1 to 365
      
      # Update current_data with the predicted price and maintain feature names
      new_data = current_data.copy()
      new_data.iloc[0, :-1] = current_data.iloc[0, 1:]  # Shift data to the left
      new_data.iloc[0, -1] = next_price[0]  # Update with new prediction
      
      current_data = new_data
  
  # Create a DataFrame for future prices
  future_prices_df = pd.DataFrame({
      'ID': ids,  # Changed column name here
      'Prediction': future_prices
  })
  
  # Save the future prices DataFrame to a CSV file
  future_prices_df.to_csv('submission_v2.csv', index=False)
  
  # Display the future prices table
  print(future_prices_df)
  
  # Plot the predicted sugar prices
  plt.figure(figsize=(14, 7))
  plt.plot(future_prices_df['ID'], future_prices_df['Prediction'], marker='o', linestyle='-', color='b')  # Changed column name here
  plt.title('Predicted Sugar Prices for Next 365 Days')
  plt.xlabel('ID')  # Changed label here
  plt.ylabel('Prediction')
  plt.grid(True)
  plt.xticks(rotation=45)
  plt.tight_layout()  # Adjust layout to fit labels
  plt.show()

- ✅ ฟีเจอร์ที่ 2

## Installation 🛠️
```bash
git clone https://github.com/username/repo.git
cd repo
