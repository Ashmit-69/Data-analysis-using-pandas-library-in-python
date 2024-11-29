import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import random
data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\CSV File\Top10VideoGameStocks.csv")
print(data.describe())
print(data.info())
dummies = pd.get_dummies(data,columns=['Ticker Symbol'],dtype='int64')
print(dummies)
data1 = dummies.drop(columns=['Date'])
print(data1)
print(data1.columns)
#print(data1.corr().to_string())
X = data1 [['Open','High','Low','Close','Adj Close']]
Y = data1 ['Volume']
print(X)
print(Y)
random.seed(1)
X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size= .30)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
print(knn.score(X_test, Y_test))
print("accuracy = {}".format(round(knn.score(X_test,Y_test),2)*100)+"%")
