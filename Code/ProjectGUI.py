
import sys

from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal
from sklearn.cluster import KMeans
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#::--------------------------------
# Default font size for all the windows
#::--------------------------------
font_size_window = 'font-size:15px'

class EDAGraphs(QMainWindow):
    #::---------------------------------------------------------
    # This class creates a canvas with a plot to show the EDA results for each variable

    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Crate a canvas with the layout
        #::--------------------------------------------------------
        super(EDAGraphs, self).__init__()

        self.Title = "EDA plots"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)


        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(["Sales percentage by year", "Monthly number of sales for 2016", "Monthly number of sales for 2017", "Top 10 customers by quantity bought",
             "Number of customers by country (excluding UK)", "Monthly total expense distribution"])

        self.dropdown1.currentIndexChanged.connect(self.update)
        self.label = QLabel("A plot:")

        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(QLabel("Select the plot"))
        self.layout.addWidget(self.dropdown1)
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)
        self.show()
        self.update()

    def update(self):
        #::--------------------------------------------------------
        # This method executes each time a change is made on the canvas.
        #::--------------------------------------------------------
        self.ax1.clear()
        cat1 = self.dropdown1.currentText()

        if cat1 == 'Sales percentage by year':
            years = df['Year'].value_counts()
            labels = ['2017', '2016']
            colors = ['#ff9999', '#ffcc99']
            self.ax1.pie(x=years, autopct='%.1f%%', explode=[0.05] * 2, labels=labels, pctdistance=0.5, colors=colors)
            self.ax1.set_title('Sales percentage by year')
            plt.show()


        if cat1 == 'Monthly number of sales for 2016':
            sales_16 = df[df['Year'] == 2016]
            monthly_16 = sales_16['Month'].value_counts().sort_index()
            self.ax1.bar(monthly_16.index.tolist(), monthly_16.values, color='orange')
            self.ax1.set_xticks(np.arange(11, 13, 1))
            self.ax1.set_title('Monthly number of sales for 2016')
            self.ax1.set_xlabel('Month')
            self.ax1.set_ylabel('number of sales')
            self.ax1.set_aspect('auto')
            plt.show()

        if cat1 == 'Monthly number of sales for 2017':
            sales_17 = df[df['Year'] == 2017]
            monthly_17 = sales_17['Month'].value_counts().sort_index()
            self.ax1.bar(monthly_17.index.tolist(), monthly_17.values, color='orange')
            self.ax1.set_xticks(np.arange(1, 13, 1))
            self.ax1.set_title('Monthly number of sales for 2017')
            self.ax1.set_xlabel('Month')
            self.ax1.set_ylabel('number of sales')
            self.ax1.set_aspect('auto')
            plt.show()

        if cat1 == 'Top 10 customers by quantity bought':
            df['CustomerID'] = df['CustomerID'].astype(str)
            customers = df.groupby('CustomerID')['Quantity'].sum()
            top_customers = customers.sort_values(ascending=False).head(10)
            labels = top_customers.index.tolist()
            values = top_customers.values
            self.ax1.bar(labels, values, color='indigo')
            self.ax1.set_title('Top 10 customers by quantity bought')
            self.ax1.set_xlabel('Customer ID')
            self.ax1.tick_params(axis='x', labelrotation=45)
            self.ax1.set_ylabel('Quantity')
            self.ax1.set_aspect('auto')
            plt.show()

        if cat1 == 'Number of customers by country (excluding UK)':
            countries = df['Country'].value_counts()[1:]
            self.ax1.bar(countries.index, countries, color='red')
            self.ax1.tick_params(axis='x', labelrotation=90)
            self.ax1.set_title('Number of customers by country (excluding UK)')
            self.ax1.set_aspect('auto')
            plt.show()

        if cat1 == 'Monthly total expense distribution':
            sns.barplot('Month', 'TotalExpense', data=df, hue='Year', ax = self.ax1)
            self.ax1.set_title("Monthly TotalExpense distribution")
            self.ax1.set_aspect('auto')
            plt.show()

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


class ClusteringKMeans(QMainWindow):
    #::---------------------------------------------------------
    # This class crates a canvas with a plot the results of KMeans algorithm
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Crate a canvas with the layout
        #::--------------------------------------------------------
        super(ClusteringKMeans, self).__init__()

        self.Title = "Cluster plots"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)


        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(["Total Purchases vs Total Expenditure", "Avg. value per purchase vs Total Expenditure", "No. of unique items vs Total Expenditure",
             "Avg. value per purchase vs Total Purchases", "No. of unique items vs Total Purchases", "No. of unique items vs Avg. value per purchase"])

        self.dropdown1.currentIndexChanged.connect(self.update)
        self.label = QLabel("A plot:")

        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(QLabel("Select the plot"))
        self.layout.addWidget(self.dropdown1)
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)
        self.show()
        self.update()

    def update(self):
        #::--------------------------------------------------------
        # This method executes each time a change is made on the canvas
        #::--------------------------------------------------------
        self.ax1.clear()
        cat1 = self.dropdown1.currentText()

        if cat1 == 'Total Purchases vs Total Expenditure':
            sns.scatterplot(x=ar[0], y=ar[1], hue='Cluster', data=cluster_df1, palette='tab10', ax = self.ax1)
            plt.show()

        if cat1 == 'Avg. value per purchase vs Total Expenditure':
            sns.scatterplot(x=ar[0], y=ar[2], hue='Cluster', data=cluster_df1, palette='tab10', ax=self.ax1)
            plt.show()

        if cat1 == 'No. of unique items vs Total Expenditure':
            sns.scatterplot(x=ar[0], y=ar[3], hue='Cluster', data=cluster_df1, palette='tab10', ax=self.ax1)
            plt.show()

        if cat1 == 'Avg. value per purchase vs Total Purchases':
            sns.scatterplot(x=ar[1], y=ar[2], hue='Cluster', data=cluster_df1, palette='tab10', ax=self.ax1)
            plt.show()

        if cat1 == 'No. of unique items vs Total Purchases':
            sns.scatterplot(x=ar[1], y=ar[3], hue='Cluster', data=cluster_df1, palette='tab10', ax=self.ax1)
            plt.show()

        if cat1 == 'No. of unique items vs Avg. value per purchase':
            sns.scatterplot(x=ar[2], y=ar[3], hue='Cluster', data=cluster_df1, palette='tab10', ax=self.ax1)
            plt.show()

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

class ClusteringBoxPlots(QMainWindow):
    #::---------------------------------------------------------
    # This class crates a canvas with a plot to show boxplots for the clusters
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Crate a canvas with the layout
        #::--------------------------------------------------------
        super(ClusteringBoxPlots, self).__init__()

        self.Title = "Cluster Box plots"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)


        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(['Total Expenditure', 'Total Purchases', 'Avg. value per purchase', 'No. of unique items'])

        self.dropdown1.currentIndexChanged.connect(self.update)
        self.label = QLabel("A plot:")

        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(QLabel("Select feature to plot"))
        self.layout.addWidget(self.dropdown1)
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)
        self.show()
        self.update()

    def update(self):
        #::--------------------------------------------------------
        # This method executes each time a change is made on the canvas
        #::--------------------------------------------------------
        self.ax1.clear()
        cat1 = self.dropdown1.currentText()

        if cat1 == 'Total Expenditure':
            cluster_df1.boxplot(column=ar[0], by=["Cluster"], ax=self.ax1)
            plt.show()

        if cat1 == 'Total Purchases':
            cluster_df1.boxplot(column=ar[1], by=["Cluster"], ax=self.ax1)
            plt.show()

        if cat1 == 'Avg. value per purchase':
            cluster_df1.boxplot(column=ar[2], by=["Cluster"], ax=self.ax1)
            plt.show()

        if cat1 == 'No. of unique items':
            cluster_df1.boxplot(column=ar[3], by=["Cluster"], ax=self.ax1)
            plt.show()

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

class PlotCanvas(FigureCanvas):
    #::----------------------------------------------------------
    # creates a figure on the canvas
    # later on this element will be used to draw a histogram graph
    #::----------------------------------------------------------
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self):
        self.ax = self.figure.add_subplot(111)

class CanvasWindow(QMainWindow):
    #::----------------------------------
    # Creates a canvaas containing the plot for the initial analysis
    #;;----------------------------------
    def __init__(self, parent=None):
        super(CanvasWindow, self).__init__(parent)

        self.left = 200
        self.top = 200
        self.Title = 'Elbow method'
        self.width = 750
        self.height = 750
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=7, height=6)
        self.m.move(0, 30)

class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.Title = 'High Value Customer Identification'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the menu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #::-----------------------------
        # Create the menu bar
        # and three items for the menu, File, EDA Analysis and ML Models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color: lightblue')

        fileMenu = mainMenu.addMenu('File')
        EDAMenu = mainMenu.addMenu('EDA Analysis')
        ClusteringMenu = mainMenu.addMenu('K Means Clustering')

        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)


        EDAButton = QAction(QIcon('analysis.png'), 'Useful plots', self)
        EDAButton.setStatusTip('Final Happiness Graph')
        EDAButton.triggered.connect(self.EDA)
        EDAMenu.addAction(EDAButton)

        Clustering1Button = QAction(QIcon('analysis.png'), 'Calculating best K', self)
        Clustering1Button.setStatusTip('Presents the initial datasets')
        Clustering1Button.triggered.connect(self.Clustering1)
        ClusteringMenu.addAction(Clustering1Button)

        Clustering2Button = QAction(QIcon('analysis.png'), 'Clustering', self)
        Clustering2Button.setStatusTip('cluster')
        Clustering2Button.triggered.connect(self.Clustering2)
        ClusteringMenu.addAction(Clustering2Button)

        Clustering3Button = QAction(QIcon('analysis.png'), 'Cluster boxplots', self)
        Clustering3Button.setStatusTip('cluster_boxplot')
        Clustering3Button.triggered.connect(self.Clustering3)
        ClusteringMenu.addAction(Clustering3Button)

        self.dialogs = list()

    def EDA(self):
        #::---------------------------------------------------------
        # This function creates an instance of EDAGraphs class
        # This class creates a graph using the features in the dataset
        # happiness vrs the score of happiness
        #::---------------------------------------------------------
        dialog = EDAGraphs()
        self.dialogs.append(dialog)
        dialog.show()

    def Clustering1(self):
        #::------------------------------------------------------
        # Creates the histogram
        # The X variable contains the happiness.score
        # X was populated in the method data_happiness()
        # at the start of the application
        #::------------------------------------------------------
        sum_sqEr = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, init='random', n_init=10, max_iter=10)
            kmeans.fit(X)
            sum_sqEr.append(kmeans.inertia_)

        dialog = CanvasWindow(self)
        dialog.m.plot()
        dialog.m.ax.plot(range(1, 11), sum_sqEr)
        dialog.m.ax.set_xticks(range(1, 11))
        dialog.m.ax.set_xlabel("Number of Clusters \n We select k = 4 using Elbow method")
        dialog.m.ax.set_ylabel("Sum of Squared Errors")
        dialog.m.draw()
        self.dialogs.append(dialog)
        dialog.show()

    def Clustering2(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the happiness dataset
        #::-------------------------------------------------------------
        dialog = ClusteringKMeans()
        self.dialogs.append(dialog)
        dialog.show()

    def Clustering3(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the happiness dataset
        #::-------------------------------------------------------------
        dialog = ClusteringBoxPlots()
        self.dialogs.append(dialog)
        dialog.show()

def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    ex.show()
    sys.exit(app.exec_())

def customers():
    #::--------------------------------------------------
    # Loads the dataset 2017.csv ( Index of happiness and esplanatory variables original dataset)
    # Loads the dataset final_happiness_dataset (index of happiness
    # and explanatory variables which are already preprocessed)
    # Populates X,y that are used in the classes above
    #::--------------------------------------------------
    global df
    global X
    global ar
    global cluster_df1
    df = pd.read_csv('Ecommerce.csv', parse_dates=['InvoiceDate'], encoding='unicode_escape')
    df.drop(['Unnamed: 8'], axis=1, inplace=True)
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df.drop(['InvoiceDate'], axis=1, inplace=True)
    df['TotalExpense'] = df['Quantity'] * df['UnitPrice']
    df.dropna(axis=0, inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)
    df = df.drop_duplicates(subset=['InvoiceNo', 'CustomerID', 'Description', 'Quantity'], keep='first')

    transactions = df[['InvoiceNo', 'CustomerID']]
    transactions = transactions.drop_duplicates()
    transactions = transactions.groupby(by='CustomerID', as_index=False).count()
    transactions = transactions.rename(columns={'InvoiceNo': 'Total Purchases'})

    unique_items = df[['StockCode', 'CustomerID']]
    unique_items = unique_items.drop_duplicates()
    unique_items = unique_items.groupby(by='CustomerID', as_index=False).count()
    unique_items = unique_items.rename(columns={'StockCode': 'No. of unique items'})

    customer_stat = df.groupby(['CustomerID'])['TotalExpense'].agg([np.sum, np.mean, np.max, np.min])
    df1 = pd.DataFrame(customer_stat)
    df1.columns = ['Total Expenditure', 'MeanAmt', 'MaxAmt', 'MinAmt']

    cluster_df1 = df1[['Total Expenditure']].merge(transactions, how='left', on='CustomerID')
    cluster_df1['Avg. value per purchase'] = cluster_df1['Total Expenditure'] / cluster_df1['Total Purchases']
    cluster_df1 = cluster_df1.merge(unique_items, how='left', on='CustomerID')
    X = cluster_df1[['Total Expenditure', 'Total Purchases', 'Avg. value per purchase', 'No. of unique items']]
    ar = ['Total Expenditure', 'Total Purchases', 'Avg. value per purchase', 'No. of unique items']

    kmeans = KMeans(n_clusters=4, random_state=42, init='random', n_init=10, max_iter=10)
    kmeans.fit(X)
    cluster_df1['Cluster'] = kmeans.predict(X)


if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    customers()
    main()