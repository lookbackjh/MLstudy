import numpy as np
import matplotlib.pyplot as plt 
from gradient import *
import pandas
import csv
## Data Generation

index=np.arange(0,101,1)
N = 100
theta = np.array([[1], [3]])
X = np.c_[np.random.rand(N,1), np.ones((N,1))]
y = np.dot(X,theta)+ 0.3*np.random.randn(N,1)
true_theta = np.linalg.lstsq(X, y, rcond=None)[0]

plt.legend()
plt.scatter(X[:,0],y)
y=np.squeeze(y)## making y from 2d to 1d
g_theta_hist,g_loss_hist, _ =gradDescent(X,y,alpha=0.01,num_iter=50,backtracking=False)
b_loss_hist,b_theta_hist = minibatchgradDescent(X,y,batchsize=1,alpha=0.01,num_iter=50) ##setting batchsize=1-> SGD

index=np.arange(0,50,1)
#plt.plot(index,g_loss_hist[0:50],label=r"$\alpha$={}".format(0.01))
plt.plot(index,np.mean(b_loss_hist,axis=1),label=r"sgd")
plt.legend()