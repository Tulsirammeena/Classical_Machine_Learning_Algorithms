
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Data_train = pd.read_csv("A2Data_train.csv",header = None)
Data_test = pd.read_csv("A2Data_test.csv", header = None)
X = Data_train.iloc[:,:-1]
y = Data_train.iloc[:,-1].to_frame()
X_test = Data_test.iloc[:,:-1]
y_test = Data_test.iloc[:,-1].to_frame()
X_test.head()
y.shape

"""#Q1(a): Closed form solution """

X=X.to_numpy()
y=y.to_numpy()
X_test=X_test.to_numpy()
X_test=X_test.transpose()
X=X.transpose()
y_test=y_test.to_numpy()
print(X.shape)
W_ml = np.linalg.inv(X @ X.transpose()) @ X @y
print(y.shape)

W_ml.shape

print(X_test.shape)
def Sq_error(X, w, y):
  Y_er_T = X.transpose() @ w
  Y_er = Y_er_T - y
  kr = Y_er.transpose()
  er = kr @ Y_er
  return er;
print(Sq_error(X, W_ml,y))
print(Sq_error(X_test, W_ml,y_test))

"""#Q1(b): Gradient descent algorithm"""

def plot(x,y,xlabel="X",ylabel="Y",title=""):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def gradientDecent(X,y,step):
    t=0
    iterations=[]
    errors=[]
    diffMl=[]
    np.random.seed(2)
    w=np.random.rand(100,1)
    prevError = Sq_error(X,w,y);
    prevW=w
    while(True):
        der = 2 * (((X @ X.transpose()) @ w)- (X @ y))
        w=(w - step* der)
        currError=Sq_error(X,w,y);
        if(currError[0][0] == np.inf):
            break
        iterations.append(t)
        it = t + 5
        errors.append(prevError[0][0])
        if( it < 0):
          it = it + t
        diffMl.append(np.linalg.norm(W_ml - w))
        it = 0
        diff = abs(prevError - currError)
        val = 0.00001
        if(diff <= val):
            break
        val = 0.0002
        prevW=w
        if(it < 0):
          break
        prevError=currError
        if(val < -1):
          val = 0.00001
        t+=1
    plot(iterations,errors,"iterations","Squared Error","iteration vs error")
    estimater = w
    plot(iterations,diffMl,"iterations","| Wᵗ - Wₘₗ |","iteration")

    
    return estimater


dataofX = X
labels = y
w=gradientDecent(dataofX,labels,0.0000039)    
print(Sq_error(dataofX,w,y))
print(Sq_error(X_test,w,y_test))

"""#Q1(c): Stochastic Gradient descent(Batch size = 100)"""

def stochasticGradientDecent(X,y,step,batchSize):
    t=0
    iterations=[]
    errors=[]
    diffMl=[]
    count=0
    np.random.seed(2)
    w=np.random.rand(100,1)
    s=w
    count+=1
    prevError = Sq_error(X,w,y);
    w_new=w
    prevW=w
    while(True):
        iterations.append(t)
        errors.append(prevError[0][0])
        if(count < 0):
          return
        diffMl.append(np.linalg.norm(W_ml - w_new))
        cnt = count
        indexes=np.random.randint(X.shape[1], size=batchSize)
        if(cnt < 0):
          iterations.append(t)
          errors.append(prevError[0][0])
        
        X_batch=X[:,indexes]
        y_batch=y[indexes,:]
        matmul = X_batch @ X_batch.transpose()
        mat2 = (X_batch @ y_batch)
        der_Sch = 2 * ((matmul @ w_new)- mat2)
        w_=(w_new - step* der_Sch)
        w_new = w_
        s1=s+w_
        s = s1
        count = count + 1
        val = 0.00001
        currError=Sq_error(X,w_new,y);
        vid = abs(prevError - currError)
        if(currError[0][0] == np.inf or vid <= val):

            break

        prewW=w_new
        prevError=currError
        t = t+1

    
    plot(iterations,errors,"iterations","Squared Error","iteration vs error")
    cn = 0
    plot(iterations,diffMl,"iterations","| Wᵗ - Wₘₗ |","iteration")
    cn = s/count
    w_avg = cn
    
    return cn;

#batchSize=100
#step=0.000039

w=stochasticGradientDecent(X,y,0.000039,100)
print(Sq_error(X,w,y))
print(Sq_error(X_test,w,y_test))
batchSize = 1000
w=stochasticGradientDecent(X,y,0.000039,batchSize)
print(Sq_error(X,w,y))
print(Sq_error(X_test,w,y_test))

"""#Q2: Ridge Regression Gradient descent and Cross validation"""

def gradient_descent_ridge(X, y, lambda_, alpha, max_iterations):
    m, n = X.shape
    theta = np.random.randn(n,1)
    for i in range(max_iterations):
        y_pred = np.dot(X, theta)
        error = y_pred - y
        grad = (1/m) * (np.dot(X.T, error) + lambda_*theta)
        theta = theta - alpha * grad
    return theta

X = X.transpose()
X_test = X_test.transpose()

print(X.shape)
print(y.shape)
print(X_test.shape)
print(y_test.shape)

import numpy as np
import matplotlib.pyplot as plt


X_train = X
y_train = y
X_test = X_test
y_test = y_test

def ridge_regression_gradient_descent(X, y, alpha, lambda_, n_iter):
    m, n = X.shape
    w = np.zeros((n, 1))
    
    for _ in range(n_iter):
        gradient = (2 / m) * (X.transpose() @ (X @ w - y) + lambda_ * w)
        w -= alpha * gradient
    
    return w

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Cross-validation
lambda_vals = np.logspace(-5, 5, 50)
cv_errors = []

for lambda_ in lambda_vals:
    w_ridge = ridge_regression_gradient_descent(X_train, y_train, alpha=0.001, lambda_=lambda_, n_iter=5000)
    y_pred_val = X_train @ w_ridge
    cv_error = mean_squared_error(y_train, y_pred_val)
    cv_errors.append(cv_error)

# Find best lambda and corresponding wR
best_lambda_idx = np.argmin(cv_errors)
best_lambda = lambda_vals[best_lambda_idx]
wR = ridge_regression_gradient_descent(X_train, y_train, alpha=0.001, lambda_=best_lambda, n_iter=5000)

# wML using normal equations
wML = np.linalg.inv(X_train.transpose() @ X_train) @ X_train.transpose() @ y_train

# Test errors
y_pred_ridge = X_test.transpose() @ wR
y_pred_ML = X_test.transpose() @ wML

test_error_ridge = mean_squared_error(y_test, y_pred_ridge)
test_error_ML = mean_squared_error(y_test, y_pred_ML)

print(f"Test error for wR (ridge): {test_error_ridge}")
print(f"Test error for wML (maximum likelihood): {test_error_ML}")

# Plot cross-validation error
plt.figure()
plt.plot(lambda_vals, cv_errors)
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Cross-validation error")
plt.title("Cross-validation error vs. Lambda")
plt.show()