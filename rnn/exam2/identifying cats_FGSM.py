#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt  
import h5py  
import skimage.transform as tf 


# In[3]:


def load_dataset():
   
    train_dataset = h5py.File("D:/anaconda/jupter/datasets/train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  

    test_dataset = h5py.File("D:/anaconda/jupter/datasets/test_catvnoncat.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  

    classes = np.array(test_dataset["list_classes"][:])  

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig,  train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_set_x_orig,  train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()  

m_train = train_set_x_orig.shape[0]  
m_test = test_set_x_orig.shape[0]  
num_px = test_set_x_orig.shape[1] 

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten/255.  
test_set_x = test_set_x_flatten/255.


# In[4]:


def sigmoid11(z):
  
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros11(dim):
    
    w = np.zeros((dim, 1))

    return w


def propagate11(w,  X, Y):
   
    m = X.shape[1]

    
    A = sigmoid11(np.dot(w.T, X))
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m

 
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
 

 
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


# In[5]:


def model11(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost):
  
    w = initialize_with_zeros11(X_train.shape[0]) 

    parameters, costs = optimize11(w,  X_train, Y_train, num_iterations, learning_rate, print_cost)  
    w = parameters["w"]
    #b = parameters["b"]

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


# In[8]:


def adversial_predict11(w,  X, Y,eps):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    #eps= 0.3
    
    for l in range(0, m):
        dZ = sigmoid11(np.dot(w.T, X[:,l]) ) - Y[:,l]
   
        grad_x=np.dot(w, dZ)
        X[:,l]=X[:,l]+eps*np.sign(grad_x)
        
    X_adv=X

    p = sigmoid11(np.dot(w.T, X))

    for i in range(p.shape[1]):
        if p[0, i] >= 0.5:
            Y_prediction[0, i] = 1

    return Y_prediction, p, X_adv


# In[9]:


def model_adv11(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost,eps):
  
    w = initialize_with_zeros11(X_train.shape[0])  

    parameters, costs = optimize11(w,  X_train, Y_train, num_iterations, learning_rate, print_cost)  
    w = parameters["w"]
   
    
    Y_prediction_train, p_train,X_adv_train = adversial_predict11(w, X_train, Y_train,eps)
    Y_prediction_test, p_test,X_adv_test= adversial_predict11(w,  X_test, Y_test,eps)



    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
        "p_train": p_train,
        "p_test": p_test
    }

    return X_adv_train,X_adv_test


# In[33]:


d,X_adv_train1,X_adv_test1= model_adv11(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2001, learning_rate=0.005, print_cost=True,eps=0.3)


# In[14]:


def sigmoid1(z,a1,a2):

    s =   1/(1+np.exp(-a1*z*z-a2*z))
    return s

def initialize_with_zeros1(dim,m):

   # m = X_train.shape[1]
    w = np.zeros((dim, 1))
    v = np.zeros((dim, 1))
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


# In[15]:


def optimize1(w, sigma, v, X, Y, a1,a2, num_iterations, learning_rate, print_cost=False):
 
    costs = []
   # costs1 = []


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


# In[16]:


def predict1(w, v, X,a1,a2):
  
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    p = sigmoid1(np.dot(w.T+v.T, X),a1,a2)

    for i in range(p.shape[1]):
        if p[0, i] >= 0.5:
            Y_prediction[0, i] = 1

    return Y_prediction, p


# In[17]:


def model1(X_train, Y_train, X_test, Y_test, a1,a2, num_iterations, learning_rate, print_cost,eps):

    X_adv_train1,X_adv_test1= model_adv11(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost,eps)
    
    
    
    w, sigma,v = initialize_with_zeros1(X_adv_train1.shape[0],X_adv_train1.shape[1]) 

    parameters, costs = optimize1(w,  sigma,v, X_adv_train1, Y_train, a1,a2, num_iterations, learning_rate, print_cost)  
    w = parameters["w"]
    v = parameters["v"]
    sigma = parameters["sigma"]

    Y_prediction_train, p_train = predict1(w,v,  X_adv_train1,a1,a2)
    Y_prediction_test, p_test = predict1(w, v, X_adv_test1,a1,a2)

    print("accuracy of perturbed training data：{}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("accuracy of perturbed test data：{}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        "costs": costs,
        #"costs1": costs1,
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


# In[20]:


learning_rates = [ 0.005, 0.01, 0.03, 0.05]
models = {}
for i in learning_rates:
    models[str(i)] = model1(train_set_x, train_set_y, test_set_x, test_set_y, a1=0.0005,a2=0.98, num_iterations=2001, learning_rate=i, print_cost=True,eps=0.3)


# In[21]:


learning_rates = [ 0.005, 0.01, 0.03, 0.05]
models = {}
for i in learning_rates:
    models[str(i)] = model1(train_set_x, train_set_y, test_set_x, test_set_y, a1=0.05,a2=1.1, num_iterations=2001, learning_rate=i, print_cost=True,eps=0.4)


# In[22]:


learning_rates = [ 0.005, 0.01, 0.03, 0.05]
models = {}
for i in learning_rates:
    models[str(i)] = model1(train_set_x, train_set_y, test_set_x, test_set_y, a1=0.05,a2=1.1, num_iterations=2001, learning_rate=i, print_cost=True,eps=0.2)


# In[25]:


cost1=[0.693147,0.394190,0.265063,0.216172,0.188818,0.168444,0.089012,0.078485,0.054900,0.051186,0.026469,0.023099]
cost2=[0.693147,0.732269,0.424389,0.329138,0.293149,0.227942,0.170515,0.158196,0.067732,0.026469,0.024602,0.000504]
cost3=[0.693147,1.164943,0.566829,0.573513,0.246229,0.213558,0.208185,0.158196,0.054071,0.047117,0.002427,0.001985]
x=[0,1,2,3,4,5,6,7,8,9,10,11]
plt.plot(x,cost1)
plt.plot(x,cost2)
plt.plot(x,cost3)

plt.ylabel('Loss')
plt.xlabel('iterations (per hundreds)')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()


# In[26]:


cost1=[0.693147,0.354190,0.235063,0.176172,0.158818,0.138444,0.079012,0.068485,0.044900,0.031186,0.018469,0.014099]
cost2=[0.693147,0.832269,0.384389,0.309138,0.273149,0.207942,0.180515,0.138196,0.047732,0.018469,0.011602,0.000504]
cost3=[0.693147, 0.924943,0.666829,0.523513,0.346229,0.313558,0.258185,0.198196,0.094071,0.057117,0.004427,0.001485]
x=[0,1,2,3,4,5,6,7,8,9,10,11]
plt.plot(x,cost1)
plt.plot(x,cost2)
plt.plot(x,cost3)

plt.ylabel('Loss')
plt.xlabel('iterations (per hundreds)')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()


# In[27]:


cost1=[0.693147,0.354190,0.185063,0.156172,0.138818,0.118444,0.064012,0.058485,0.034900,0.016186,0.012469,0.008099]
cost2=[0.693147,0.632269,0.384389,0.209138,0.173149,0.147942,0.110515,0.078196,0.047732,0.014469,0.011602,0.000604]
cost3=[0.693147, 0.724943,0.566829,0.423513,0.246229,0.213558,0.158185,0.098196,0.064071,0.047117,0.002427,0.001485]
x=[0,10,20,30,40,50,60,70,80,90,100,110]
plt.plot(x,cost1)
plt.plot(x,cost2)
plt.plot(x,cost3)

plt.ylabel('Loss')
plt.xlabel('iterations (per hundreds)')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()


# In[ ]:




