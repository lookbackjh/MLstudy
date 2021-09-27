import numpy as np
import matplotlib.pyplot as plt 
from gradient import *
import pandas
import csv
## Data Generation
n=100
X=np.hstack([np.ones((n,1)),np.random.rand(n,1)]) ### Design Matrix For X
##print(X)
theta=np.array([4,3])
Y=np.dot(X,theta)+np.random.normal(0,1,n)
#plt.scatter(X[:,1],Y)
##scattered image of X and Y.. Now Lets do the optimization
df=pandas.read_csv("data.csv")

y=df['y']
X=df.loc[:,df.columns!='y']
X_train,X_test,y_train,y_test=splitData(X,y)
X_train=featurescale(X_train)
X_test=featurescale(X_test)
theta_init=np.random.rand(X_train.shape[1])
print(theta_init.shape)
loss=SquareLossfunction(X_train,y_train,theta_init)
print(loss)
num_iter=150
theta_hist,loss_hist=gradDescent(X_train,y_train,num_iter=num_iter)
print(loss_hist)