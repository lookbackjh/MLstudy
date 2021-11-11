import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def featurescale(X_train,X_test):
    ## before doing the gradient, the data must be feature scaled.., 
    ## feature이 여러가지일때, 특정 feature의 범위는 gradient 를 시행할때, 큰 영향을 줄수있다. i.e 키, 몸무게가 feature 인 경우.
    ## 따라서, feature의 크기를 어느정도 조절해주는 도구가 필요한데, 이게 feature Scaling이다.
    ## feature scaling을 함에 있어서, gradient descent의 수렴속도가 훨씬 빨라질 수 잇다. 
    ## feature scaling은 min-max scaling과 standard normal scaling이 있는데, 두가지 모두 자주 사용된다. 
    ## 두가지 방법 은 쓰임세가 살짝 다르긴한데, 대부분의 상황에서 통용될수 있는 min-max scaling을 사용하도록 하겠다.
    scalar=MinMaxScaler()
    scalar.fit(X_train)
    X_train=scalar.transform(X_train)
    X_test=scalar.transform(X_test)
    ##주의!!!: testset의 Scaling은 train set 의 스케일링과 동일하게 진행.
    return X_train,X_test
def splitData(X,y):
    ## 검증을 위해 데이터 X를 train 데이터와 test data로 분리시킴, 통상적으로 약 7:3 언저리 의 비율로 가름
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
    return X_train,X_test,y_train,y_test

def SquareLossfunction(X,y,theta, l2_reg=0):
    ## Square Loss 
    ## X: n* d matrix  : n=number of data, d=number of feature
    ## Y: n*1 matrix: n= number of data
    ## theta: 1* d matrix, d: number of feature
    ## loss =avg(X*thata.T-Y+ l2_reg*(L2norm(theta)))
    m=X.shape[0]
    loss_term=np.mean(np.square((np.dot(X,theta)-y)))
    reg_term=np.linalg.norm(theta)*l2_reg
    loss=loss_term+reg_term
    return loss
def computegrad(X,y,theta,l2_reg=0):
    ## X: n* d matrix  : n=number of data, d=number of feature
    ## Y: n*1 matrix: n= number of data
    m=X.shape[0]
    temp=np.dot(X,theta)-y
    grad_term=(2.0/m)*(np.dot(X.T,temp))
    reg_term=2*l2_reg*theta
    return grad_term +reg_term 
def gradDescent(X,y,alpha=0.1,num_iter=1000,backtracking=True, l2_reg=0.01): 
    ## the very basic gradient descent..
    ##things to consider.. initialization of theta, iteration..
    feat_num=X.shape[1]
    data_num=X.shape[0]

    theta_hist=np.zeros((num_iter+1,feat_num)) ## storing the historical data of theta
    loss_hist = np.zeros(num_iter+1) ## storing loss value to see if gradient is doing well..
    theta_init=np.random.rand(feat_num) ## 통상적으로 0,1 사이에 랜덤하게 생성
    theta_hist[0,:]=theta_init
    loss_hist[0]=SquareLossfunction(X,y,theta_hist[0,:]) ##이니셜
    bactracknum=0
    ## initialize theta with random number between 0 and 1
    if backtracking:
        alp, beta=0.3,0.9
        for i in range(0,num_iter):
            cur_theta=theta_hist[i,:]
            dx=-computegrad(X,y,cur_theta)
            t=1
            while True:
            
                n_loss=SquareLossfunction(X,y,cur_theta+t*dx)
                o_loss=SquareLossfunction(X,y,cur_theta)-alp*np.dot(dx,dx)*t
                if n_loss>o_loss:
                    t=0.9*t
                    bactracknum+=1
                else:
                    break
            theta_hist[i+1]=cur_theta-t*computegrad(X,y,cur_theta)
            loss_hist[i+1]=SquareLossfunction(X,y,theta_hist[i+1,:])

    else:
        for i in range(0,num_iter):
            cur_theta=theta_hist[i,:]
            theta_hist[i+1,:]=cur_theta-alpha*computegrad(X,y,cur_theta,l2_reg)
            loss_hist[i+1]=SquareLossfunction(X,y,theta_hist[i+1,:],l2_reg)
    return theta_hist, loss_hist,bactracknum

def minibatchgradDescent(X,y,  batchsize=1,alpha=0.005, num_iter=100):
    ##things to Consider
    ## Batchsize, doing Bactracking or not ,iteration, epoch.. 
    ## sequence 1. separating batches..
    n=X.shape[0]
    num_feat=X.shape[1]
    if n%batchsize==0:
        b_num=(int)(n/batchsize)
    else: b_num=(int)(n/batchsize)+1
    b_index=np.arange(0,b_num,1)
    theta_hist=np.zeros((num_iter,b_num,num_feat)) ## storing the historical data of theta
    loss_hist = np.zeros((num_iter,b_num)) ## storing loss value to see if gradient is doing well..
    theta_init=np.random.rand(num_feat)
    random.shuffle(b_index)
    theta_hist[0,0,:]=theta_init
    for j in range(num_iter):
        for i,index in enumerate(b_index):
            cur_theta=theta_hist[j,i,:]
            if index==batchsize-1:
                splitX=X[index*batchsize:n,:]
                splity=y[index*batchsize:n]
            else:
                splitX=X[index*batchsize:(index+1)*batchsize,:]
                splity=y[index*batchsize:(index+1)*batchsize]
            if i+1<b_num:
                theta_hist[j,i+1,:]=cur_theta-alpha*computegrad(splitX,splity,cur_theta)
                loss_hist[j,i]=SquareLossfunction(X,y,cur_theta)
            else :
                if j+1==num_iter:
                    return loss_hist,theta_hist
                theta_hist[j+1,0,:]=cur_theta-alpha*computegrad(splitX,splity,cur_theta)
                loss_hist[j+1,i]=SquareLossfunction(X,y,cur_theta)
        random.shuffle(b_index)

    return loss_hist, theta_hist

