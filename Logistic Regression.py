import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
#sns.scatterplot(data=data1)
#plt.plot(X,Y)
random.seed(1)
X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size= .30)
model= LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print('model_score',model.score(X_train,Y_train))
print('Accuracy:{:.2f}%'.format(accuracy * 100))
