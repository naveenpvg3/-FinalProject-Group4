import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np

df = pd.read_csv('Ecommerce.csv', parse_dates=['InvoiceDate'], encoding= 'unicode_escape')

#%%
print("PLotting monthly TotalExpense distribution")
plt.figure(figsize=(12, 8))
sns.barplot('Month', 'TotalExpense', data=df, hue='Year')
plt.title("Monthly TotalExpense distribution")
plt.show()
print('#',50*"-")

#%%
print("Calculating TotalExpense for each CustomerID")
customer_stat = df.groupby(['CustomerID'])['TotalExpense'].agg([np.sum, np.mean, np.max, np.min])
df1 = pd.DataFrame(customer_stat)
df1.columns = ['Total Expenditure', 'MeanAmt', 'MaxAmt', 'MinAmt']
print(df1.head(5))
print('#',50*"-")

#%%
print("Calculating number of invoices for each CustomerID")
transactions = df[['InvoiceNo', 'CustomerID']]
transactions = transactions.drop_duplicates()
transactions = transactions.groupby(by='CustomerID', as_index=False).count()
transactions = transactions.rename(columns={'InvoiceNo': 'Total Purchases'})
print(transactions.head(5))
print('#',50*"-")

#%%
print("Calculating number of unique items purchased by each customer")
unique_items = df[['StockCode', 'CustomerID']]
unique_items = unique_items.drop_duplicates()
unique_items = unique_items.groupby(by='CustomerID', as_index=False).count()
unique_items = unique_items.rename(columns={'StockCode': 'No. of unique items'})
print(unique_items.head(5))
print('#',50*"-")
