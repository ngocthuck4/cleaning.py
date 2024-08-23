import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Tạo dữ liệu giả lập cho các bảng
transactions = pd.DataFrame({
    'TransactionID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'CustomerID': [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'ProductID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
    'Date': pd.to_datetime([
        '2024-01-01', '2024-01-15', '2024-02-01', '2024-03-01', '2024-03-15', '2024-04-01',
        '2024-05-01', '2024-06-01', '2024-07-01', '2024-08-01', '2024-09-01', '2024-10-01'
    ]),
    'Quantity': [1, 2, 1, 3, 2, 1, 1, 2, 1, 3, 2, 1],
    'UnitPrice': [200, 220, 210, 230, 240, 250, 260, 270, 280, 290, 300, 310],
    'TotalPrice': [200, 440, 210, 690, 480, 250, 260, 540, 280, 870, 600, 310],
    'PaymentMethod': ['Credit Card', 'Debit Card', 'Credit Card', 'PayPal', 'Credit Card', 'Debit Card',
                       'Credit Card', 'Debit Card', 'PayPal', 'Credit Card', 'Debit Card', 'PayPal'],
    'OrderStatus': ['Completed', 'Completed', 'Completed', 'Pending', 'Completed', 'Shipped',
                    'Completed', 'Completed', 'Pending', 'Shipped', 'Completed', 'Completed'],
    'ShippingAddress': ['123 Main St', '123 Main St', '456 Elm St', '789 Maple St', '101 Oak St', '202 Pine St',
                         '303 Cedar St', '404 Birch St', '505 Willow St', '606 Spruce St', '707 Pine St', '808 Maple St'],
    'DiscountApplied': [10, 20, 0, 15, 5, 0, 10, 20, 5, 15, 0, 10]
})

# Tính doanh số bán hàng theo tháng
monthly_sales = transactions.groupby(transactions['Date'].dt.to_period('M')).agg({
    'TotalPrice': 'sum'
}).reset_index()

monthly_sales['Month'] = monthly_sales['Date'].dt.month
monthly_sales['Year'] = monthly_sales['Date'].dt.year

# Lấy các cột cần thiết
monthly_sales = monthly_sales[['Month', 'TotalPrice']]
monthly_sales.columns = ['Month', 'TotalSales']

# Đảm bảo không có giá trị âm
monthly_sales['TotalSales'] = monthly_sales['TotalSales'].clip(lower=0)

# Tạo các đặc trưng và biến mục tiêu
X = monthly_sales[['Month']]  # Đặc trưng: Tháng
y = monthly_sales['TotalSales']  # Biến mục tiêu: Doanh số bán hàng

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Kiểm tra số lượng mẫu sau khi chia dữ liệu
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Vẽ đồ thị so sánh giữa doanh số thực tế và dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Sales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Fitted Line')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.grid(True)

# Hiển thị đồ thị
plt.show()

# Dự đoán doanh số trong các tháng tiếp theo
future_months = pd.DataFrame({'Month': [13, 14, 15]})
future_sales_predictions = model.predict(future_months)

# Đảm bảo không có giá trị âm trong dự đoán
future_sales_predictions = np.clip(future_sales_predictions, a_min=0, a_max=None)

print("Predicted sales for upcoming months:", future_sales_predictions)
plt.savefig('sales_plot.png')  # Lưu đồ thị dưới dạng tệp PNG


