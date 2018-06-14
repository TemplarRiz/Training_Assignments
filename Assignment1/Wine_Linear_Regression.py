import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#--------------Helper Functions--------------------

def getdataset():
  dataset = pd.read_csv("/home/abhishek/Downloads/winequality-red.csv",delimiter = ';')
  dataset = np.array(dataset)
  return dataset

def getdata() :
  dataset = getdataset()
  p = dataset.shape[0]
  X = np.array(dataset[:,:11])
  Y = np.zeros((dataset.shape[0],1))
  Y[:,0] = dataset[:,11]
  return train_test_split(X,Y,train_size = 0.8,test_size = 0.2)  # Data divided into train and test in the ratio of 80:20

def initialize(X1,N) :
  m = X1.shape[0] 
  att = X1.shape[1]
  cost = np.ones((N,1))
  xcor = np.ones((N,1))
  return (np.zeros((att,1)) , 0 , cost , xcor)

def Linear_Regression_Forward(X1,W,b):
  return np.dot(X1,W) + b

def compute_cost(Z,Y1):
  m = Y1.shape[0]
  J = np.dot((Z-Y1).T,Z-Y1) / m
  return J[0,0]

def Linear_Regression_Backward(Z,Y1,X1):
  m = X1.shape[0]
  dZ = Z - Y1
  dW = np.dot(X1.T,dZ) / m
  db = np.sum(dZ) / m
  return dZ,dW,db

def Linear_Regression_Update(W,b,al,dW,db):
  W = W - al*dW
  b = b - al*db
  return (W,b)

def Plot(xcor,cost):
  plt.plot(xcor,cost)
  plt.show()

#-----------------------Main Function----------------------------


N = 1000                                        # number of iterations
al = 0.00045                                    # learning rate
X1,X2,Y1,Y2 = getdata()
W,b,cost,xcor = initialize(X1,N)

for i in range(N):
  Z = Linear_Regression_Forward(X1,W,b)
  J = compute_cost(Z,Y1)
  dZ,dW,db = Linear_Regression_Backward(Z,Y1,X1)
  W,b = Linear_Regression_Update(W,b,al,dW,db)
  
  
  cost[i] = J
  xcor[i] = i


Plot(xcor,cost)                                #Use for plotting J vs iteration

print("Cost for training dataset is :" , J)


Z = Linear_Regression_Forward(X2,W,b)
J = compute_cost(Z,Y2)  
print("Cost for test dataset is :" , J)























