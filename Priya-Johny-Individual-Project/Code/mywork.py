#%% md
#cite:Kaggle
# Importing Data

#%%
#importing python libraries
import pandas as pd #cleaning noisy dataset
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from PyQt5.QtWidgets import QApplication
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

#%%
#reading acquired dataset
df = pd.read_csv('Ecommerce.csv', parse_dates=['InvoiceDate'], encoding= 'unicode_escape')
print("The original dataset has", len(df), "observations and", len(df.columns), "variables/features. \n")
print("Name of all the variables:")
print(df.columns, '\n')
df.columns=['InvoiceNo',
                'StockCode',
                'Description',
                'Quantity',
                'InvoiceDate',
                'UnitPrice',
                'CustomerID',
                'Country', ' ']

#%%
print('Exploring the head (first few rows)and tail (last few rows) of the dataset\n')
print('Head:\n')
print(df.head())

print('Tail:\n')
print(df.tail())

print('#',50*"-")
#%%
print("Datatype of all the variables:\n")
df.info()

print('4 of the variables are object type, quantity variable is in integer type,\n '
      'date is in datetime64 and rest are in float datatype. \n'
      'This will be helpful for analysing if the dataset can be used in our models')
print('#',50*"-")
#%%
print("Statistical Summary :\n")
print(df.describe())
print('Statistical summary give us an overview of the statistics such as mean, count, minumum,\n'
      ' maximum, percentiles and standard deviation values of each variable.')

print('#',50*"-")
#%%
print("Number of null data-points in each variable:\n")
print(df.isnull().sum())
print('Variable description has over 1454 null values and CustomerId has 135080 null values.')

print('#',50*"-")
#%%
print("Number of unique data-points in each variable:\n")

print(df.nunique())

print('.nunique() gives us the unique values present in each variable and \n'
      ' we can separate based on single variable.\n'
      ' This unique values help us group values with least unique values together in a single group')

print('Display of unique values of type categorical in case-of large datasets\n')

for col_name in df.columns:
    if df[col_name].dtype == 'object':
        unique_cat = len(df[col_name].unique())
        print("categorical feature '{col_name}' has {unique_cat}  unique categories".format(col_name=col_name,unique_cat=unique_cat))

print('#',50*"-")
#%%
print("Data Cleaning \n")

#%%
print("Dropping the last empty variable Unnamed: 8")
df.drop([' '], axis=1, inplace=True)

print('#',50*"-")

#%%
print("Dropping rows with missing/na values")
df.dropna(axis = 0, inplace = True)

print('#',50*"-")

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

#%%
print("Extracting year, month and date from the InvoiceDate variable to \n")
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df.drop(['InvoiceDate'], axis=1, inplace=True)
print('A new column named Year,Month and Day is created.')

#%%
print("Adding a new variable TotalExpense to the dataset")
df['TotalExpense'] = df['Quantity'] * df['UnitPrice']
print('A new column named TotalExpense is created and added to the dataset.')

#%%
print("Number of data points in the final cleaned dataset:", len(df))
print('#',50*"-")
print("Now we can save the new dataset to a csv format and use the new version for building our K-Means Clustering model\n"
      "Another method is to use GUI interactive session to save the file to csv format.")
#df.to_csv(r'df.csv', index = False)

#%% basic GUI Trial for exporting cleaned dataset to new csv, cite:datafish
'''Here I am just exploring and adding a gui where the clean dataset created can be saved through an interactive session\n.
Here the user can just save the data in csv format after datacleaning'''
#import tkinter packages
import tkinter as tk
from tkinter import filedialog
from pandas import DataFrame

#read df ->cleaned dataset
df = DataFrame(df)
root = tk.Tk()

GUI = tk.Canvas(root, width=300, height=300, bg='black', relief='raised')
GUI.pack()

#exportCSV() function is used to convert existing to a new csv file
def exportCSV():
    global df
    export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')
    df.to_csv(export_file_path, index=False, header=True)
#button
Button_CSV = tk.Button(text='Export CSV', command=exportCSV, bg='blue', fg='black',
                             font=('helvetica', 12, 'bold'))
GUI.create_window(150, 150, window=Button_CSV)
root.mainloop()




