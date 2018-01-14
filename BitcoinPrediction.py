#Abdirahman Mohamed
'''
A Program that predicts the price of Bitcoin using Linear,
Polynomial and Radio Basis Function modules. Written in Python 2.7
'''
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
    """Reads the CSV file and get the date and price"""
    try:
        csvfile= open(filename,"r")
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('/')[0])) # get the date and remove the /
            prices.append(float(row[1])) # get the price
    except ValueError:
        pass
    return
def predict_prices(dates,prices,x):
    """Draws the prediction models on the screen"""
    dates= np.reshape(dates,(len(dates),1))
    svr_lin = SVR(kernel='linear',C=1e3) # Linear model features
    svr_poly = SVR(kernel='poly',C=1e3,degree=2) # Polynomial model features
    svr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1) # RBF model features
    svr_lin.fit(dates,prices)
    svr_poly.fit(dates,prices)
    svr_rbf.fit(dates,prices)
    plt.scatter(dates,prices,color='black',label='Data')
    plt.plot(dates,svr_rbf.predict(dates),color='red',label='RBF Model')
    plt.plot(dates,svr_lin.predict(dates),color='green',label='Linear Model')
    plt.plot(dates,svr_poly.predict(dates),color='blue',label='Polynomial Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Bitcoin Price Prediction')
    plt.legend()
    plt.show() # Display it in a graph
    return svr_rbf.predict(x)[0],svr_lin.predict(x)[0],svr_poly.predict(x)[0]
get_data('BCHARTS.csv') # The csv file
predicted_prices = predict_prices(dates,prices,14) # predicted prices for Jan 14
print predicted_prices

