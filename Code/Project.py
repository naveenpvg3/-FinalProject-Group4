#%% md

# Importing Data

#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

#%%

df = pd.read_csv('Ecommerce.csv', parse_dates=['InvoiceDate'], encoding= 'unicode_escape')
print("The original dataset has", len(df), "observations and", len(df.columns), "variables. \n")
print("Name of all the variables:")
print(df.columns, '\n')

print(df.head())
print('#',50*"-")

#%%
print("Datatype of all the variables:")
df.info()
print('#',50*"-")

#%%
print("Summary of the dataset:")
print(df.describe())
print('#',50*"-")

#%%
print("Number of null data-points in each variable:")
print(df.isnull().sum())
print('#',50*"-")

#%%
print("Number of unique data-points in each variable:")
print(df.nunique())
print('#',50*"-")

#%%

print("Data Cleaning \n")

#%%
print("Dropping the variable Unnamed: 8")
df.drop(['Unnamed: 8'], axis=1, inplace=True)

#%%
print("Extracting year, month and date from the InvoiceDate variable")
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df.drop(['InvoiceDate'], axis=1, inplace=True)

#%%
print("Adding a new variable TotalExpense to the dataset")
df['TotalExpense'] = df['Quantity'] * df['UnitPrice']

#%%
print("Dropping rows with missing/na values")
df.dropna(axis = 0, inplace = True)

#%%
print("Changing the datatype of CustomerID to int")
df['CustomerID'] = df['CustomerID'].astype(int)
print('#',50*"-")

#%%
print("Rechecking for null values:")
print(df.isnull().sum())
print('#',50*"-")

#%%
print("Checking for duplicates \n")
df = df.drop_duplicates(subset =['InvoiceNo', 'CustomerID', 'Description', 'Quantity'], keep = 'first')
print(df.info(),'\n')
print("Number of data points in the final cleaned dataset:", len(df))
print('#',50*"-")

#%%
print("EXPLORATORY DATA ANALYSIS: \n")

#%%
print("Plotting Year wise number of sales")
years = df['Year'].value_counts()
print(years)
pie, ax = plt.subplots(figsize=[7,5])
labels = ['2017', '2016']
colors = ['#ff9999', '#ffcc99']
plt.pie(x = years, autopct='%.1f%%', explode=[0.05]*2, labels=labels, pctdistance=0.5, colors = colors)
plt.title('Sales percentage by year')
print("Plotting pie chart")
plt.show()
print('#',50*"-")

#%%
print("Plotting monthly sales by year")
sales_16 = df[df['Year'] == 2016]
sales_17 = df[df['Year'] == 2017]
monthly_16 = sales_16['Month'].value_counts()
monthly_17 = sales_17['Month'].value_counts()

plt.figure(figsize=(8,5))
monthly_16.sort_index().plot(kind='bar', color='orange')
plt.title('Monthly number of sales for 2016')
plt.xlabel('Month')
plt.ylabel('number of sales')
plt.grid()
plt.show()

plt.figure(figsize=(8,5))
monthly_17.sort_index().plot(kind='bar', color='blue')
plt.title('Monthly number of sales for 2017')
plt.xlabel('Month')
plt.ylabel('number of sales')
plt.grid()
plt.show()
print('#',50*"-")

#%%
customers = df.groupby('CustomerID')['Quantity'].sum()

print("Plotting top 10 customers after summing their purchase quantities")
top_customers = customers.sort_values(ascending=False).head(10)

plt.figure(figsize=(8,8))
top_customers.plot(kind='bar', color='indigo')
plt.title('Top 10 customers by quantity bought')
plt.xlabel('Customer ID')
plt.ylabel('Quantity')
plt.show()
print('#',50*"-")

#%%
print("Country wise number of sales:")
print(df['Country'].value_counts().head())

#%%
countries = df['Country'].value_counts()[1:]
fig, ax = plt.subplots(figsize = (12,10))

ax.bar(countries.index, countries, color= 'purple')
ax.set_xticklabels(countries.index, rotation = 90)
ax.set_title('Number of customers by country (excluding UK)')
print("Plotting country wise sales")
plt.show()
print('#',50*"-")

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

#%%
print("CLUSTERING:\n")

print("We identified the metrics to use for clustering as follows: For each customer ID, we calculated TotalExpense, "
      "number of purchases, average amount per purchase, number of unique items bought. With the help of these metrics"
      "we will be able to categorize the customers into groups: High spending customers, regular (loyal) customers, etc. \n")

cluster_df1 = df1[['Total Expenditure']].merge(transactions, how='left', on='CustomerID')
cluster_df1['Avg. value per purchase'] = cluster_df1['Total Expenditure']/cluster_df1['Total Purchases']
cluster_df1 = cluster_df1.merge(unique_items, how='left', on='CustomerID')

print(cluster_df1.head())

X = cluster_df1[['Total Expenditure', 'Total Purchases', 'Avg. value per purchase','No. of unique items']]
ar= ['Total Expenditure', 'Total Purchases', 'Avg. value per purchase','No. of unique items']

print("Plotting SSE against various values of k")
sum_sqEr = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, init='random', n_init=10, max_iter=10)
    kmeans.fit(X)
    sum_sqEr.append(kmeans.inertia_)

f, axes = plt.subplots(1,1,figsize=(8,8))
plt.plot(range(1, 11), sum_sqEr)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Errors")
plt.show()

print("Using Elbow method, we can decide k = 4 as an appropriate number of clusters.")

kmeans = KMeans(n_clusters=4, random_state=42, init='random', n_init=10, max_iter=10)
kmeans.fit(X)
cluster_df1['Cluster']=kmeans.predict(X)

for i in range(3):
    for j in range(i+1,4):
        sns.scatterplot(x=ar[i], y=ar[j], hue='Cluster', data=cluster_df1, palette='tab10')
        plt.show()

print("From the scatter plots showing different clusters, we can conclude a few things and assign some characteristics to the clusters, \n")
print("Cluster 0: Total expenditure is the least of all, total purchases are minimal and minimal number of unique items are bought by them.")
print("Cluster 1: Highest expenditure is observed with a wide range of values. Total purchases are moderate. These customers have bought large number of unique items.")
print("Cluster 2: Total expenditure is the same as cluster 0 and everything else is at a minimal level.")
print("Cluster 3: Total expenditure is minimal but everything else is moderate in numbers. \n")

for i in range(4):
      cluster_df1.boxplot(column=ar[i], by=["Cluster"])
      plt.show()

print("We plotted boxplots to confirm our observations from the scatter plots. \n")
print("Based on these observations we can categorize these clusters as follows: \n")
print("Cluster 0 and 2 need to be focussed more on, in terms of discounts and other offers in order to increase their purchase numbers.")
print("Cluster 1 can be considered as high-value customers for whom a loyalty program should be rolled out. These customers are loyal as well seeing the number of purchases.")
print("Cluster 3 consists of regular value but loyal customers who visit the store pretty often.")


