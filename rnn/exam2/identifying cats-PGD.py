#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt  
import h5py  #
import skimage.transform as tf  # 


# In[2]:


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


# In[6]:


def sigmoid11(z):
 
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros11(dim):
   
    w = np.zeros((dim, 1))
    #b = 0
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


# In[7]:


def model11(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost):
  
    w = initialize_with_zeros11(X_train.shape[0])  # 

    parameters, costs = optimize11(w,  X_train, Y_train, num_iterations, learning_rate, print_cost)  # 
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


d,w = model11(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2001, learning_rate=0.005, print_cost=True)


# In[17]:


def sigmoid1(z,a1,a2):
   
    s =   1/(1+np.exp(-a1*z*z-a2*z))
    return s

def initialize_with_zeros1(dim,m):

   # m = X_train.shape[1]
    w = np.zeros((dim, 1))
    v = np.zeros((dim, 1))

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


# In[18]:


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


# In[19]:


def predict1(w, v, X,a1,a2):
   
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    p = sigmoid1(np.dot(w.T+v.T, X),a1,a2)

    for i in range(p.shape[1]):
        if p[0, i] >= 0.5:
            Y_prediction[0, i] = 1

    return Y_prediction, p


# In[20]:


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


# In[21]:


def model_adv_pgd(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost,eps):
 
    w = initialize_with_zeros11(X_train.shape[0]) 

    parameters, costs = optimize11(w,  X_train, Y_train, num_iterations, learning_rate, print_cost)  
    w = parameters["w"]
    #b = parameters["b"]
    
    Y_prediction_train, p_train,X_adv_train = adversial_predict_pgd(w, X_train, Y_train,eps)
    Y_prediction_test, p_test,X_adv_test= adversial_predict_pgd(w,  X_test, Y_test,eps)



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


# In[13]:


X_adv_train1,X_adv_test1= model_adv_pgd(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2001, learning_rate=0.005, print_cost=True,eps=0.3)


# In[22]:


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


# In[25]:


learning_rates = [ 0.005, 0.01, 0.03, 0.05,0.08,0.1]
models = {}
for i in learning_rates:
    models[str(i)] = model2(train_set_x, train_set_y, test_set_x, test_set_y, a1=0.05,a2=1.1, num_iterations=2001, learning_rate=i, print_cost=True,eps=0.3)


# In[ ]:


learning_rates = [ 0.005, 0.01, 0.03, 0.05,0.08,0.1]
models = {}
for i in learning_rates:
    models[str(i)] = model2(train_set_x, train_set_y, test_set_x, test_set_y, a1=0.05,a2=1.1, num_iterations=2001, learning_rate=i, print_cost=True,eps=0.2)


# In[ ]:


learning_rates = [ 0.005, 0.01, 0.03, 0.05,0.08,0.1]
models = {}
for i in learning_rates:
    models[str(i)] = model2(train_set_x, train_set_y, test_set_x, test_set_y, a1=0.05,a2=1.1, num_iterations=2001, learning_rate=i, print_cost=True,eps=0.4)


# In[3]:


cost2=[0.693147,0.781318,0.411435,0.370170,0.289182,0.239689,0.191985,0.154121,0.047117,0.023560,0.007799,0.003828]
x=[0,100,200,300,400,500,600,700,800,900,1000,1100]

plt.plot(x,cost2)
plt.ylabel('cost')
plt.xlabel('iterations')

plt.show()


# In[4]:


cost2=[0.693147,0.699439,0.431435,0.370170,0.249182,0.209689,0.121985,0.069124,0.043861,0.027243,0.013108,0.011350]
x=[0,100,200,300,400,500,600,700,800,900,1000,1100]
plt.plot(x,cost2)
plt.ylabel('cost')
plt.xlabel('iterations')
#plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# In[6]:


cost2=[0.693147,0.599439,0.451435,0.330170,0.259182,0.189689,0.101985,0.08944,0.055418,0.032927,0.016337,0.002264]
x=[0,100,200,300,400,500,600,700,800,900,1000,1100]

plt.plot(x,cost2)
plt.ylabel('cost')
plt.xlabel('iterations')
#plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# In[29]:


cost1=[0.693147,0.599439,0.451435,0.330170,0.259182,0.189689,0.101985,0.08944,0.055418,0.032927,0.016337,0.002264]
cost2=[0.693147,0.619439,0.341435,0.280170,0.199182,0.129689,0.081985,0.069124,0.043861,0.027243,0.013108,0.011350]
cost3=[0.693147,0.721318,0.291435,0.240170,0.129182,0.08689,0.051985,0.024121,0.009117,0.007560,0.005799,0.001828]
x=[0,1,2,3,4,5,6,7,8,9,10,11]

plt.plot(x,cost1)
plt.plot(x,cost2)
plt.plot(x,cost3)

plt.ylabel('Loss')
plt.xlabel('iterations (per hundreds)')
legend = plt.legend(loc='upper right', shadow=True)
#for i in learning_rates:
    #plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))
    
#plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# In[28]:


cost1=[0.693147,0.499439,0.381435,0.350170,0.289182,0.179689,0.111985,0.0844,0.04418,0.0092927,0.0076337,0.002264]
cost2=[0.693147,0.699439,0.351435,0.300170,0.189182,0.149689,0.071985,0.0344,0.009418,0.0062927,0.0056337,0.0002264]
cost3=[0.693147,0.781318,0.251435,0.180170,0.139182,0.089689,0.041985,0.00944,0.005418,0.0032927,0.0016337,0.003828]
x=[0,1,2,3,4,5,6,7,8,9,10,11]

plt.plot(x,cost1)
plt.plot(x,cost2)
plt.plot(x,cost3)

plt.ylabel('Loss')
plt.xlabel('iterations (per hundreds)')
legend = plt.legend(loc='upper right', shadow=True)
#for i in learning_rates:
    #plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))
    
#plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# In[27]:


cost1=[0.693147,0.599439,0.451435,0.330170,0.259182,0.149689,0.071985,0.02944,0.0085418,0.0032927,0.0016337,0.0002264]
cost2=[0.693147,0.699439,0.331435,0.170170,0.109182,0.05689,0.021985,0.009124,0.0043861,0.0027243,0.0013108,0.0011350]
cost3=[0.693147,0.711318,0.211435,0.140170,0.089182,0.049689,0.021985,0.00944,0.003418,0.0012927,0.0006337,0.0003828]
x=[0,1,2,3,4,5,6,7,8,9,10,11]

plt.plot(x,cost1)
plt.plot(x,cost2)
plt.plot(x,cost3)

plt.ylabel('Loss')
plt.xlabel('iterations (per hundreds)')
legend = plt.legend(loc='upper right', shadow=True)
#for i in learning_rates:
    #plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))
    
#plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# In[ ]:




