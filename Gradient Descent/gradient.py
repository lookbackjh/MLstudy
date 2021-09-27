import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def featurescale(X_train):
    ## before doing the gradient, the data must be feature scaled.., 
    ## feature이 여러가지일때, 특정 feature의 범위는 gradient 를 시행할때, 큰 영향을 줄수있다. i.e 키, 몸무게가 feature 인 경우.
    ## 따라서, feature의 크기를 어느정도 조절해주는 도구가 필요한데, 이게 feature Scaling이다.
    ## feature scaling을 함에 있어서, gradient descent의 수렴속도가 훨씬 빨라질 수 잇다. 
    ## feature scaling은 min-max scaling과 standard normal scaling이 있는데, 두가지 모두 자주 사용된다. 
    ## 두가지 방법 은 쓰임세가 살짝 다르긴한데, 대부분의 상황에서 통용될수 있는 min-max scaling을 사용하도록 하겠다.
    scalar=MinMaxScaler()
    data_num,feat_num=X_train.shape[0],X_train.shape[1]
    scalar.fit(X_train)
    X_train=scalar.transform(X_train)
    return X_train
def splitData(X,y):
    ## 검증을 위해 데이터 X를 train 데이터와 test data로 분리시킴, 통상적으로 약 7:3 언저리 의 비율로 가름
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
    return X_train,X_test,y_train,y_test

def SquareLossfunction(X,y,theta, l2_reg=0.01):
    ## Square Loss 
    ## X: n* d matrix  : n=number of data, d=number of feature
    ## Y: n*1 matrix: n= number of data
    ## theta: 1* d matrix, d: number of feature
    ## loss =avg(X*thata.T-Y+ l2_reg*(L2norm(theta)))
    loss_term=np.mean(np.square((np.dot(X,theta)-y)))
    #reg_term=np.linalg.norm(theta)
    loss=loss_term
    return loss
def computegrad(X,y,theta,l2_reg=0.01):
    m=X.shape[0]
    temp=np.dot(X,theta)-y
    grad_term=2.0*np.mean((np.dot(X.T,temp)))
    return grad_term  
def gradDescent(X,y,alpha=0.05,num_iter=1000): 
    ## the very basic gradient descent..
    ##things to consider.. initialization of theta, iteration..
    feat_num=X.shape[1]
    data_num=X.shape[0]

    theta_hist=np.zeros((num_iter+1,feat_num)) ## storing the historical data of theta
    loss_hist = np.zeros(num_iter+1) ## storing loss value to see if gradient is doing well..
    
    ## initialize theta with random number between 0 and 1
    theta_init=np.random.rand(feat_num)
    theta_hist[0,:]=theta_init
    loss_hist[0]=SquareLossfunction(X,y,theta_hist[0,:])
    for i in range(0,num_iter):
        cur_theta=theta_hist[i,:]
        theta_hist[i+1,:]=cur_theta-alpha*computegrad(X,y,cur_theta)
        loss_hist[i+1]=SquareLossfunction(X,y,theta_hist[i+1,:])
    return theta_hist, loss_hist

def regularizedgradDescent(X,y,theta):
    ##things to Consider
    ## Batchsize, doing Bactracking or not ,iteration, epoch.. 
    return 0





