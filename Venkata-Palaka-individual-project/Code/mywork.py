
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


import sys
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        self.setCentralWidget(sc)

        self.show()


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()
