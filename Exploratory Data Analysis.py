import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset (update file path to your local file)
file_path_cleaned = '/Users/gabby/Desktop/Python and code/Pricing Opt/Cleaned_superstore.csv'
df_user_cleaned = pd.read_csv(file_path_cleaned)

# Step 1: Understand the Data
print("Dataset Info:")
print(df_user_cleaned.info())

print("\nFirst 5 Rows:")
print(df_user_cleaned.head())

print("\nSummary Statistics:")
print(df_user_cleaned.describe())

# Step 2: Analyze the Relationship Between Price and Quantity
plt.figure(figsize=(8, 6))
plt.scatter(df_user_cleaned["Price"], df_user_cleaned["Quantity"], alpha=0.5)
plt.title("Price vs. Quantity")
plt.xlabel("Price")
plt.ylabel("Quantity Sold")
plt.grid()
plt.show()

# Step 3: Check the Impact of Discounts
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_user_cleaned["Discount"], y=df_user_cleaned["Quantity"])
plt.title("Discount vs. Quantity")
plt.xlabel("Discount")
plt.ylabel("Quantity Sold")
plt.grid()
plt.show()

# Group by Discount and calculate average Quantity
discount_analysis = df_user_cleaned.groupby("Discount")["Quantity"].mean().reset_index()
print("\nAverage Quantity Sold by Discount Level:")
print(discount_analysis)

# Step 4: Analyze Time-Based Trends
# Convert Order.Date to datetime if not already done
df_user_cleaned["Order.Date"] = pd.to_datetime(df_user_cleaned["Order.Date"], errors='coerce')

# Aggregate sales by date
daily_sales = df_user_cleaned.groupby("Order.Date")["Sales"].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(daily_sales["Order.Date"], daily_sales["Sales"])
plt.title("Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid()
plt.show()
