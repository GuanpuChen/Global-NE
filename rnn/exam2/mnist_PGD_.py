#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import scipy.io as si
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[3]:


mnist = si.loadmat('D:/anaconda/jupter/mnist/MNIST_data/MNIST.mat') 


# In[4]:


X1, y1 = mnist["img_train"], mnist["label_train"]
X2, y2 = mnist["img_t10k"], mnist["label_t10k"]
y1 = y1.astype(np.uint8)
y2 = y2.astype(np.uint8)
X_train, X_test, y_train, y_test = X1[:200], X2[:50], y1[:200], y2[:50]
y_train_0 = (y_train == 0)  
y_test_0 = (y_test == 0)
train_set_x_flatten = X_train.reshape(X_train.shape[0], -1).T  
test_set_x_flatten = X_test.reshape(X_test.shape[0], -1).T

train_set_x = train_set_x_flatten/255.  
test_set_x = test_set_x_flatten/255.
train_set_y=y_train_0.T
test_set_y=y_test_0.T


# In[5]:


def sigmoid11(z):

    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros11(dim):
   
    w = np.zeros((dim, 1))
    #b = 0
    return w


def propagate11(w,  X, Y):
  
    m = X.shape[1]

    #
    A = sigmoid11(np.dot(w.T, X))
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m

    # 
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m


    # 
    grads = {
        "dw": dw
    }

    return grads, cost


def optimize11(w,  X, Y, num_iterations, learning_rate, print_cost=False):

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate11(w, X, Y)

        dw = grads["dw"]
      

     
        w = w - learning_rate * dw
   

      

    params = {
        "w": w
    }

    return params, costs


def predict11(w, X):
 
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    p = sigmoid11(np.dot(w.T, X))

    for i in range(p.shape[1]):
        if p[0, i] >= 0.5:
            Y_prediction[0, i] = 1

    return Y_prediction, p


# In[6]:


def model11(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost):
  
    w = initialize_with_zeros11(X_train.shape[0])  # 

    parameters, costs = optimize11(w,  X_train, Y_train, num_iterations, learning_rate, print_cost)  # 
    w = parameters["w"]
   

    Y_prediction_train, p_train = predict11(w, X_train)
    Y_prediction_test, p_test = predict11(w,  X_test)

    print("accuracy of training data：{}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("accuracy of test data：{}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
      #  "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
        "p_train": p_train,
        "p_test": p_test
    }

    return d,w


# In[7]:


d,w = model11(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2001, learning_rate=0.005, print_cost=True)


# In[8]:


def sigmoid1(z,a1,a2):

    s =   1/(1+np.exp(-a1*z*z-a2*z))
    return s

def initialize_with_zeros1(dim,m):

   # m = X_train.shape[1]
    w = 5*np.ones((dim, 1))
    v = 5*np.ones((dim, 1))
    #b = 0
    sigma=0.1*np.ones((m, 1))
    return w, sigma,v


def propagate1(w, v, X, Y,sigma,a1,a2):
    
    m = X.shape[1]
  
    z = np.dot(w.T+v.T, X) 
   
    A = sigmoid1(z,a1,a2)
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    lambda1=0.1
    lambda2=0.1
 
    dw  = 1/m* np.dot(X,-(2*a1*z.T*sigma+a2*sigma))+1/m*a2* np.dot(X,(np.ones(m)-Y).T)+1/m*2*a1* np.dot(X,(np.ones(m)-Y).T*z.T)+lambda1*w
    dv  = 1/m* np.dot(X,-(2*a1*z.T*sigma+a2*sigma))+1/m*a2* np.dot(X,(np.ones(m)-Y).T)+1/m*2*a1* np.dot(X,(np.ones(m)-Y).T*z.T)-lambda2*v

    dsigma=1/m *((-a1*z*z-a2*z).T-np.log(sigma/(1-sigma)))


    grads = {'dw':dw, 'dv':dv, 'dsigma':dsigma}

    return grads, cost


# In[9]:


def optimize1(w, sigma, v, X, Y, a1,a2, num_iterations, learning_rate, print_cost=False):
 
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate1(w,v,  X, Y,sigma,a1,a2)

        dw = grads["dw"]
        dv = grads["dv"]
        dsigma = grads['dsigma']

       
        w = w - learning_rate * dw
        v = v + learning_rate * dv
        sigma = sigma + learning_rate * dsigma
        
        for l in range(0, len(v)):
            if v[l]<-0.2: 
                v[l]=-0.2
            elif v[l]>0.2: 
                v[l]=0.2
                
        for j in range(0, len(sigma)):
            if sigma[j]<0.005: 
                sigma[j]=0.005
            elif sigma[j]>0.99: 
                sigma[j]=0.99
        

     
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("The cost after  %d times is：%f" % (i, cost))

    params = {
        "w": w,
        "v": v,
        "sigma":sigma
    }

    return params, costs


# In[10]:


def predict1(w, v, X,a1,a2):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    p = sigmoid1(np.dot(w.T+v.T, X),a1,a2)

    for i in range(p.shape[1]):
        if p[0, i] >= 0.5:
            Y_prediction[0, i] = 1

    return Y_prediction, p


# In[11]:


def adversial_predict_pgd(w,  X, Y,eps):

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    MaxIter = 40
    step_size = 1e-2
    #eps= 0.3
    
    for l in range(0, m):
        x_original=X[:,l]
        x_adv=X[:,l]
        for k in range(MaxIter):
            dZ = sigmoid11(np.dot(w.T, X[:,l]) ) - Y[:,l]
        #dw = np.dot(X, dZ.T) / m
            grad_x=np.dot(w, dZ)
            x_adv=x_adv+step_size*np.sign(grad_x)
            for q in range(0,len(x_adv)):
                t=x_adv[q] - x_original[q]
                x_adv[q]=x_original[q] + max(-eps, min(t, eps))
                x_adv[q]=max(0, min(x_adv[q],1))
        X[:,l]=x_adv
        
    X_adv=X

    p = sigmoid11(np.dot(w.T, X))

    for i in range(p.shape[1]):
        if p[0, i] >= 0.5:
            Y_prediction[0, i] = 1

    return Y_prediction, p, X_adv


# In[12]:


def model_adv_pgd(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost,eps):

    w = initialize_with_zeros11(X_train.shape[0]) 

    parameters, costs = optimize11(w,  X_train, Y_train, num_iterations, learning_rate, print_cost)  
    w = parameters["w"]
    #b = parameters["b"]
    
    Y_prediction_train, p_train,X_adv_train = adversial_predict_pgd(w, X_train, Y_train,eps)
    Y_prediction_test, p_test,X_adv_test= adversial_predict_pgd(w,  X_test, Y_test,eps)

    #Y_prediction_train, p_train = predict(w, b, X_train)
    #Y_prediction_test, p_test = predict(w, b, X_test)

    print("accuracy of perturbed training data：{}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("accuracy of perturbed test data：{}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        #"b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
        "p_train": p_train,
        "p_test": p_test
    }

    return X_adv_train,X_adv_test


# In[14]:


def model2(X_train, Y_train, X_test, Y_test, a1,a2, num_iterations, learning_rate, print_cost,eps):

    
    X_adv_train1,X_adv_test1= model_adv_pgd(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost,eps)
    
    
    
    w, sigma,v = initialize_with_zeros1(X_adv_train1.shape[0],X_adv_train1.shape[1]) 

    parameters, costs = optimize1(w,  sigma,v, X_adv_train1, Y_train, a1,a2, num_iterations, learning_rate, print_cost) 
    w = parameters["w"]
    v = parameters["v"]
    sigma = parameters["sigma"]

    Y_prediction_train, p_train = predict1(w,v,  X_adv_train1,a1,a2)
    Y_prediction_test, p_test = predict1(w, v, X_adv_test1,a1,a2)

   

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "v": v,
        "sigma": sigma,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
        "p_train": p_train,
        "p_test": p_test
    }

    return d


# In[24]:


learning_rates = [  0.05, 0.1,0.15]
models = {}
for i in learning_rates:
    models[str(i)] = model2(train_set_x, train_set_y, test_set_x, test_set_y, a1=0.0005,a2=0.98, num_iterations=3001, learning_rate=i, print_cost=True,eps=0.4)

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('Loss')
plt.xlabel('iterations (per hundreds)')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()


# In[17]:


learning_rates = [  0.05, 0.1,0.15]
models = {}
for i in learning_rates:
    models[str(i)] = model2(train_set_x, train_set_y, test_set_x, test_set_y, a1=0.0005,a2=0.98, num_iterations=3001, learning_rate=i, print_cost=True,eps=0.27)

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('Loss')
plt.xlabel('iterations (per hundreds)')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()


# In[23]:


learning_rates = [  0.05, 0.1,0.15]
models = {}
for i in learning_rates:
    models[str(i)] = model2(train_set_x, train_set_y, test_set_x, test_set_y, a1=0.0005,a2=0.98, num_iterations=3001, learning_rate=i, print_cost=True,eps=0.2)

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('Loss')
plt.xlabel('iterations (per hundreds)')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()


# In[ ]:




