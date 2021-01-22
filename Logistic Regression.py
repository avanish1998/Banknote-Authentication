#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[66]:


def sigmoid(z):
    z=1/(1+np.exp(-z))
    return z   


# In[67]:


def signum(z):
    if z>0:
        return 1
    elif z<0:
        return -1
    else:
        return 0


# In[68]:


dataset = pd.read_csv("banknotes.csv")
dataset.shape


# In[69]:


sns.pairplot(dataset)


# In[70]:


features = dataset.iloc[:,0:4].values
print(features)
labels = dataset.iloc[:,4].values
labels.reshape(labels.shape[0],1)


# In[71]:


# min-max normalization
row,col = features.shape
mins = np.zeros(shape=(col),dtype=np.float32)
maxs = np.zeros(shape=(col),dtype=np.float32)
for j in range(col):
    mins[j] = np.min(features[:,j])
    maxs[j] = np.max(features[:,j])
X = np.copy(features)
for i in range(row):
    for j in range(col):
        X[i,j]=(features[i,j]-mins[j])/(maxs[j]-mins[j])
print(X)


# In[72]:


# train-test split -> 80-20
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.2)
X_train=X_train.T
X_test=X_test.T


# In[180]:


# initialization
dim = X_train.shape[0]
w = np.random.randn(dim,1)*0.01
print(w)
b=0


# In[181]:


# propagation 
m = X_train.shape[1]
iteration=[]
loss=[]
num_of_iterations = 20000
learning_rate =0.5
for i in range(num_of_iterations):
    iteration.append(i)
    A=sigmoid(np.matmul(w.T,X_train)+b)
    cost = -(1/m)*(np.sum((Y_train*np.log(A))+((1-Y_train)*np.log(1-A))))
    loss.append(cost)
    dw = (1/m)*(np.matmul(X_train, (A-Y_train).T))
    db = (1/m)*np.sum(A-Y_train)
    w = w-learning_rate*dw
    b = b-learning_rate*db


# In[182]:


#prediction
print(": Logistic Regression with no regularization : ")
m = X_test.shape[1]
Y_predict = np.zeros((1,m))
A = sigmoid(np.matmul(w.T,X_test)+b)
for i in range(A.shape[1]):
    if A[0,i]>=0.5:
        Y_predict[0,i]=1
    else:
        Y_predict[0,i]=0
correct = 0
print("\tTest accuracy: {} %".format(100 - np.mean(np.abs(Y_predict - Y_test)) * 100))
Y_test.reshape(1,275)
tp=0
fp=0
fn=0
for i in range(Y_predict.shape[1]):
    if Y_predict[0,i] == 1 and Y_test[i] == 1:
        tp += 1
    elif Y_predict[0,i] == 1 and Y_test[i] == 0:
        fp += 1
    elif Y_predict[0,i] == 0 and Y_test[i] == 1:
        fn += 1
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f = 2*precision*recall/(precision+recall)
print(": F-score : ",f)

plt.plot(iteration,loss)
plt.xlabel('No. of epochs')
plt.ylabel('Loss')
plt.title("Learning rate : "+str(learning_rate))
plt.show()


# In[129]:


# initialization
dim = X_train.shape[0]
w = np.random.rand(dim,1)*0.01
print(w)
suma = np.sum(w,axis=0,keepdims=True)
b=0
lambd = 0.01


# In[130]:


# propagation 
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
m = X_train.shape[1]
num_of_iterations =40000
iteration=np.zeros(num_of_iterations)
loss=np.zeros(num_of_iterations)
learning_rate =0.5
for i in range(num_of_iterations):
    iteration[i]=i
    A=sigmoid(np.matmul(w.T,X_train)+b)
    suma = np.sum(w,axis=0,keepdims=True)
    cost = -(1/m)*(np.sum((Y_train*np.log(A))+((1-Y_train)*np.log(1-A))))+lambd*np.sum(np.abs(w),axis=0)
    loss[i]=cost
    dw = (1/m)*(np.matmul(X_train, (A-Y_train).T))
    db = (1/m)*np.sum(A-Y_train)
    w = w-learning_rate*dw-learning_rate*(lambd/2)*signum(suma)
    b = b-learning_rate*db
print(" W : ",w)
print(" b : ",b)


# In[131]:


#prediction
print(": Logistic Regression with L1 regularization : ")
m = X_test.shape[1]
Y_predict = np.zeros((1,m))
A = sigmoid(np.matmul(w.T,X_test)+b)
for i in range(A.shape[1]):
    if A[0,i]>=0.5:
        Y_predict[0,i]=1
    else:
        Y_predict[0,i]=0
correct = 0
print("\tTest accuracy: {} %".format(100 - np.mean(np.abs(Y_predict - Y_test)) * 100))
Y_test.reshape(1,275)
tp=0
fp=0
fn=0
for i in range(Y_predict.shape[1]):
    if Y_predict[0,i] == 1 and Y_test[i] == 1:
        tp += 1
    elif Y_predict[0,i] == 1 and Y_test[i] == 0:
        fp += 1
    elif Y_predict[0,i] == 0 and Y_test[i] == 1:
        fn += 1
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f = 2*precision*recall/(precision+recall)
print("F-score : ",f)
plt.plot(iteration,loss)
plt.xlabel('No. of epochs')
plt.ylabel('Loss')
plt.title("Learning rate : "+str(learning_rate)+"    Lambda : "+str(lambd))
plt.show()


# In[177]:


# initialization
dim = X_train.shape[0]
w = np.random.rand(dim,1)*0.01
print(w)
suma = np.sum(w,axis=0,keepdims=True)
b=0
lambd = 0.0005


# In[178]:


# propagation 
m = X_train.shape[1]
iteration=[]
loss=[]
num_of_iterations = 10000
learning_rate =1
for i in range(num_of_iterations):
    iteration.append(i)
    A=sigmoid(np.matmul(w.T,X_train)+b)
    cost = -(1/m)*(np.sum((Y_train*np.log(A))+((1-Y_train)*np.log(1-A))))+(lambd/2)*(np.sum(np.square(w),axis=0))
    loss.append(cost)
    dw = (1/m)*(np.matmul(X_train, (A-Y_train).T))
    db = (1/m)*np.sum(A-Y_train)
    for i in range(w.shape[0]):
        w[i]=w[i]-learning_rate*dw[i,0]-lambd*learning_rate*w[i]
    b = b-learning_rate*db
print(" W : ",w)
print(" b : ",b)


# In[179]:


#prediction
print(": Logistic Regression with L2 regularization : ")
m = X_test.shape[1]
Y_predict = np.zeros((1,m))
A = sigmoid(np.matmul(w.T,X_test)+b)
for i in range(A.shape[1]):
    if A[0,i]>=0.5:
        Y_predict[0,i]=1
    else:
        Y_predict[0,i]=0
correct = 0
print("\tTest accuracy: {} %".format(100 - np.mean(np.abs(Y_predict - Y_test)) * 100))
Y_test.reshape(1,275)
tp=0
fp=0
fn=0
for i in range(Y_predict.shape[1]):
    if Y_predict[0,i] == 1 and Y_test[i] == 1:
        tp += 1
    elif Y_predict[0,i] == 1 and Y_test[i] == 0:
        fp += 1
    elif Y_predict[0,i] == 0 and Y_test[i] == 1:
        fn += 1
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f = 2*precision*recall/(precision+recall)
print("F-score : ",f)
plt.plot(iteration,loss)
plt.xlabel('No. of epochs')
plt.ylabel('Loss')
plt.title("Learning rate : "+str(learning_rate)+"    Lambda : "+str(lambd))
plt.show()


# In[ ]:





# In[ ]:




