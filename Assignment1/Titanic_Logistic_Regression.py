import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#-------------------------------Helper Functions------------------------------------------

def getdataset():
  datasetfortrain = pd.read_csv("/home/abhishek/Downloads/train.csv",delimiter = ',')
  datasetfortrain = datasetfortrain.drop(['PassengerId', 'Name','Ticket','Cabin'], axis=1)
  datasetfortrain = pd.get_dummies(datasetfortrain)
  datasetfortrain = datasetfortrain.fillna(datasetfortrain.mean())
  return np.array(datasetfortrain) 

def sigmoid(Z):
  return 1/(1 + np.exp(-Z))

def getdata():
  traindataset = getdataset()
  a,b = traindataset.shape
  Y = np.zeros((a,1))
  Y[:,0] = traindataset[:,0]
  X= np.array(traindataset[:,1:])
  X1,X2,Y1,Y2 = train_test_split(X,Y,train_size = 0.8,test_size = 0.2)
  return X1.T,X2.T,Y1.T,Y2.T

def initialize():
  att,m = X1.shape
  W = np.zeros((att,1))
  b = 0
  cost = np.ones((N,1))
  xcor = np.ones((N,1))
  return W,b,xcor,cost

def Linear_Regression_Forward(X1,W,b):
  Z = np.dot(W.T,X1) + b
  return sigmoid(Z)

def compute_cost(A,Y1):
  m = Y1.shape[1]
  return -np.sum(np.multiply(Y1,np.log(A)) + np.multiply(1 - Y1,np.log(1 - A))) / m

def Linear_Regression_Backward(A,Y1,X1):
  m = X1.shape[1]
  dZ = A - Y1
  db = np.sum(dZ) / m
  dW = np.dot(X1,dZ.T) / m
  return dZ,dW,db

def Linear_Regression_Update(W,b,al,dW,db):
  W = W - al*dW
  b = b - al*db
  return W,b

def Plot(xcor,cost):
  plt.plot(xcor,cost)
  plt.show()

def accuracy(A,Y2):
  m = Y2.shape[1]
  A = np.floor(A*2)
  return np.sum(A == Y2)/m

#------------------------------------------------Main Function----------------------------------------------


N = 60000
al = 0.004
X1,X2,Y1,Y2 = getdata()
W,b,xcor,cost = initialize()

for i in range(N):
  A = Linear_Regression_Forward(X1,W,b)
  J = compute_cost(A,Y1)
  dZ,dW,db = Linear_Regression_Backward(A,Y1,X1)
  W,b = Linear_Regression_Update(W,b,al,dW,db)
  cost[i] = J
  xcor[i] = i


Plot(xcor,cost)                                #Use for plotting J vs iteration

print("Cost for training dataset is : " , J)

A = Linear_Regression_Forward(X2,W,b)
J = compute_cost(A,Y2)
print("Cost for test dataset is : " , J)
print("Accuracy for the test dataset is : " , accuracy(A,Y2))


  



















