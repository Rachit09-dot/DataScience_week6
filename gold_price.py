# ================================
# Linear Regression for Gold Price Prediction
# ================================

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load the Dataset
# Make sure the CSV file is in the same folder
df = pd.read_csv("gold_price_data.csv")

# 3. Display Dataset Info
print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# 4. Data Cleaning
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Drop rows with missing values (if any)
df.dropna(inplace=True)

# 5. Feature Selection
# Independent variables (Features)
X = df[['Open', 'High', 'Low', 'Volume']]

# Dependent variable (Target)
y = df['Close']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Create Linear Regression Model
model = LinearRegression()

# 8. Train the Model
model.fit(X_train, y_train)

# 9. Make Predictions
y_pred = model.predict(X_test)

# 10. Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error (MSE):", mse)
print("R-squared Score (R2):", r2)

# 11. Visualize Actual vs Predicted Prices
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Gold Prices")
plt.ylabel("Predicted Gold Prices")
plt.title("Actual vs Predicted Gold Price")
plt.show()

# 12. Predict a New Sample (Optional)
# Example input: Open, High, Low, Volume
sample_data = np.array([[1850, 1870, 1830, 120000]])
sample_prediction = model.predict(sample_data)

print("\nPredicted Gold Price for Sample Input:", sample_prediction[0])
