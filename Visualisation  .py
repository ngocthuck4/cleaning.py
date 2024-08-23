import pandas as pd
import matplotlib.pyplot as plt

# Giả sử bạn đã đọc các tệp CSV vào các DataFrame, ví dụ:
# customer_df = pd.read_csv('customer.csv')
# trend_df = pd.read_csv('trend.csv')
# category_df = pd.read_csv('category.csv')
# transaction_df = pd.read_csv('transaction.csv')
# product_price_df = pd.read_csv('product_price.csv')

# Ví dụ: dữ liệu mẫu cho bảng trend
trend_data = {
    'Region': ['North America', 'Europe', 'Asia', 'South America'],
    'SalesGrowthRate': [0.05, 0.03, 0.07, 0.04]
}

trend_df = pd.DataFrame(trend_data)

# Biểu đồ 1: Tỷ lệ tăng trưởng doanh số theo vùng
plt.figure(figsize=(10, 6))
plt.bar(trend_df['Region'], trend_df['SalesGrowthRate'], color='skyblue')
plt.title('Sales Growth Rate by Region')
plt.xlabel('Region')
plt.ylabel('Sales Growth Rate')
plt.savefig('sales_growth_rate_by_region.png')
plt.close()


# Bạn có thể tiếp tục với các biểu đồ khác tương tự

print("All charts have been saved successfully!")

import os
print("Current working directory:", os.getcwd())

