import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Đường dẫn đầy đủ đến tệp CSV
file_path = r'C:\Users\ASUS\PycharmProjects\pythonProject2\customer table.csv'

try:
    # Đọc tệp CSV
    df = pd.read_csv(file_path)
    print("First 5 rows of the dataframe:")
    print(df.head())
    print("\nDataframe Info:")
    print(df.info())
    print("\nDescriptive statistics:")
    print(df.describe())
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Xử lý dữ liệu bị thiếu
    df_cleaned = df.dropna()  # Xóa các hàng có giá trị bị thiếu
    df_filled = df.fillna('Unknown')  # Điền các giá trị bị thiếu bằng 'Unknown'

    # Lọc dữ liệu dựa trên cột Email
    df_cleaned = df[df['Email'].str.contains('@', na=False)]

    # Đổi tên các cột
    df.rename(columns={'FirstName': 'First_Name', 'LastName': 'Last_Name'}, inplace=True)

    # Chuyển đổi kiểu dữ liệu của các cột
    df['PostalCode'] = df['PostalCode'].astype(str)
    df['Phone'] = df['Phone'].astype(str)

    # Lưu DataFrame đã xử lý vào tệp CSV mới
    cleaned_file_path = r'C:\Users\ASUS\PycharmProjects\pythonProject2\customer_data_cleaned.csv'
    df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data has been saved to {cleaned_file_path}")

except FileNotFoundError:
    print(f"The file {file_path} does not exist. Please check the file path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")
