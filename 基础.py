from asyncio.windows_events import NULL
from re import T
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from numpy import gradient
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

class logistic:

    def __init__(self,Lambda=1,alpha=0.01,iter=2001) -> None:
        self.theta=None
        self.Lambda=Lambda
        self.alpha=alpha
        self.iter=iter

    #min cost_function
    def cost_function(self,x,y):
        m=x.shape[0]
        h_of_x=sigmoid(x*self.theta)
        return (-1.0/m)*np.sum(np.multiply(y,np.log(h_of_x))+np.multiply(1-y,np.log(1-h_of_x)))+self.Lambda/(2*m)*np.sum(np.square(self.theta))
    
    
    def fit(self,x,y):
        lossList=[]
        m=x.shape[0]
        X=np.concatenate((np.ones((m,1)),x),axis=1)
        #X=x
        n=X.shape[1]
        self.theta=np.mat(np.ones((n,1)))
        #print(self.theta.shape)
        #print(type(self.theta))
        xMat=np.mat(X)
        #print(xMat.shape)
        yMat=np.mat(y)
        yMat=yMat.T
        #print(yMat.shape)
        for i in range(self.iter):
            hx=sigmoid(np.dot(xMat,self.theta))
            gradient=1.0/m*np.dot((xMat.T),(hx-yMat))+self.Lambda*self.theta/m
            self.theta=self.theta-self.alpha*gradient
            
            if i%200==0:
                lossList.append(self.cost_function(xMat,yMat))
            
        return self.theta,lossList

    def pred(self,x):
        m=x.shape[0]
        X=np.concatenate((np.ones((m,1)),x),axis=1)
        xMat=np.mat(X)
        return np.array(sigmoid(np.dot(xMat,self.theta)))



data=pd.read_table("D:\李志鹏\python\考核\四轮\\bezdekIris.txt",header=None,sep=',')
data.columns=['sepal length','sepal width','petal length','petal width','class']
"""
min_max_scaler=preprocessing.MinMaxScaler()
data['sepal length']=min_max_scaler.fit_transform(data['sepal length'].values.reshape(-1,1))
data['sepal width']=min_max_scaler.fit_transform(data['sepal width'].values.reshape(-1,1))
data['petal length']=min_max_scaler.fit_transform(data['petal length'].values.reshape(-1,1))
data['petal width']=min_max_scaler.fit_transform(data['petal width'].values.reshape(-1,1))
print(data)
"""


x=data.drop(columns=['class'])
y=data['class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y)

lrs=[]
namelist=[]
print(y.value_counts().index)
for i in y.value_counts().index:
    print(i)
    namelist.append(i)
    list1=list(y.value_counts().index)
    list1.remove(i)
    print(list1)
    y_train1=y_train.replace(i,1)
    y_train1=y_train1.replace(list1,0)
    lr1=logistic()
    lr1.fit(x_train,y_train1)
    lrs.append(lr1)

anss=[]
for lr in lrs:
    ans=lr.pred(x_test)
    anss.append(ans)

y_pred=[]
for i in range(x_test.shape[0]):
    max=0
    index=0
    cnt=0
    for ans in anss:
        if ans[i][0]>max:
            max=ans[i][0]
            index=cnt
        cnt=cnt+1
    y_pred.append(namelist[index])

y_pre_data=pd.DataFrame(y_pred,columns=['y_predict'],index=y_test.index)
y_test_pre_data=pd.concat([y_test,y_pre_data],axis=1)
print('参数:','\n')
for lr in lrs:
    print(lr.theta,'\n')
print('结果:','\n',y_test_pre_data)
#print(y_test_pre_data.shape)
accuracy=metrics.accuracy_score(y_test, y_pred)
print('准确率:',accuracy)


#print(y_train.value_counts())
#print(y_test.value_counts())