import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge

# Tạo dữ liệu giả lập mở rộng với nhiều mẫu
np.random.seed(42)

data = {
    'ProductID': np.arange(1, 101),
    'ProductName': [f'Product{i}' for i in range(1, 101)],
    'Category': ['Electronics', 'Furniture'] * 50,
    'Price': np.random.randint(50, 300, 100),
    'StockQuantity': np.random.randint(10, 100, 100)
}

transactions = {
    'TransactionID': np.arange(1, 1001),
    'CustomerID': np.random.randint(1, 51, 1000),
    'ProductID': np.random.randint(1, 101, 1000),
    'Date': pd.date_range(start='2024-01-01', periods=1000, freq='D'),
    'Quantity': np.random.randint(1, 10, 1000),
    'UnitPrice': np.random.randint(50, 300, 1000),
    'TotalPrice': np.random.randint(50, 300, 1000) * np.random.randint(1, 10, 1000),
    'PaymentMethod': np.random.choice(['Credit Card', 'PayPal'], 1000),
    'OrderStatus': ['Completed'] * 1000,
    'ShippingAddress': [f'Address{i}' for i in range(1, 1001)],
    'DiscountApplied': np.random.randint(0, 50, 1000)
}

df_products = pd.DataFrame(data)
df_transactions = pd.DataFrame(transactions)

# Tính doanh số hàng tháng
df_transactions['Date'] = pd.to_datetime(df_transactions['Date'])
df_transactions['Month'] = df_transactions['Date'].dt.month
monthly_sales = df_transactions.groupby('Month')['TotalPrice'].sum().reset_index()

# Chuẩn bị dữ liệu cho mô hình hồi quy
months = monthly_sales[['Month']]
sales = monthly_sales['TotalPrice']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(months, sales, test_size=0.2, random_state=42)

# Xây dựng và huấn luyện mô hình hồi quy tuyến tính và hồi quy Ridge
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
sales_pred_test_lr = model_lr.predict(X_test)
sales_pred_test_ridge = model_ridge.predict(X_test)

mse_lr = mean_squared_error(y_test, sales_pred_test_lr)
r2_lr = r2_score(y_test, sales_pred_test_lr)

mse_ridge = mean_squared_error(y_test, sales_pred_test_ridge)
r2_ridge = r2_score(y_test, sales_pred_test_ridge)

print(f'Linear Regression - Mean Squared Error: {mse_lr:.2f}')
print(f'Linear Regression - R-squared: {r2_lr:.2f}')

print(f'Ridge Regression - Mean Squared Error: {mse_ridge:.2f}')
print(f'Ridge Regression - R-squared: {r2_ridge:.2f}')

# Hiển thị kết quả
plt.figure(figsize=(12, 6))
plt.scatter(months, sales, color='blue', label='Actual Sales')
plt.plot(months, model_lr.predict(months), color='green', label='Linear Regression Fit')
plt.plot(months, model_ridge.predict(months), color='red', linestyle='--', label='Ridge Regression Fit')
plt.title('Monthly Sales Prediction using Linear and Ridge Regression')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.savefig('sales_prediction_comparison.png')  # Lưu đồ thị vào file hình ảnh
plt.show()

# Dự đoán doanh số cho các tháng tiếp theo
future_months = np.array([13, 14, 15, 16, 17, 18]).reshape(-1, 1)
future_sales_pred_lr = model_lr.predict(future_months)
future_sales_pred_ridge = model_ridge.predict(future_months)

print("Linear Regression Predictions:")
for month, prediction in zip(future_months.flatten(), future_sales_pred_lr):
    print(f'Predicted sales for month {month}: {prediction:.2f}')

print("\nRidge Regression Predictions:")
for month, prediction in zip(future_months.flatten(), future_sales_pred_ridge):
    print(f'Predicted sales for month {month}: {prediction:.2f}')
plt.savefig('sales_prediction_comparison.png')
