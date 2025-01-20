import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path_cleaned = '/Users/gabby/Desktop/Python and code/Pricing Opt/Cleaned_superstore.csv'
df = pd.read_csv(file_path_cleaned)
if 'Cost' not in df.columns:
    df['Cost'] = df['Price'] * df['Quantity']

# Prepare the data for modeling
def prepare_data(df):
    X = df[['Price', 'Cost', 'Discount', 'Shipping.Cost']].values
    y = df['Quantity'].values
    return X, y

# Train the demand prediction model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    return model

# Calculate profits for a price range
def calculate_profit(price_range, cost, discount, shipping_cost, model, base_features):
    profits = []
    for price in price_range:
        # Combine price with base features
        features = [price, cost, discount, shipping_cost] + base_features
        features = np.array(features).reshape(1, -1)  
        
        # Predict quantity
        predicted_quantity = model.predict(features)[0]
        
        # Calculate profit
        profit = (price - cost - shipping_cost) * predicted_quantity
        profits.append(profit)
    return profits

# Define input features for profit calculation
X, y = prepare_data(df)
model = train_model(X, y)

# Base features
cost = 10  # Example fixed cost
discount = 0.1  # Example discount
shipping_cost = 5  # Example shipping cost
base_features = []  # Additional base features, if any

# Define price range
price_range = np.linspace(1, 50, 50)  # Example price range
profits = calculate_profit(price_range, cost, discount, shipping_cost, model, base_features)

# Finding optimal price
optimal_price = price_range[np.argmax(profits)]
max_profit = max(profits)

print(f"Optimal Price: ${optimal_price:.2f}")
print(f"Maximum Profit: ${max_profit:.2f}")

# Visualize the profit curve
plt.figure(figsize=(10, 6))
plt.plot(price_range, profits, label='Profit Curve')
plt.axvline(optimal_price, color='red', linestyle='--', label=f'Optimal Price: ${optimal_price:.2f}')
plt.xlabel('Price')
plt.ylabel('Profit')
plt.title('Profit Optimization')
plt.legend()
plt.grid(True)
plt.show()
