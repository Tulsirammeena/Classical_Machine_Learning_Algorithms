#import all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Getting data from https://huggingface.co/datasets/mnist
d0 = pd.read_csv('mnist.csv')
print(d0.head(5))
l = d0['label']
d = d0.drop("label",axis=1)
d = d/255
print(d.shape)

"""Some random Mnist data"""

plt.figure(figsize=(3,3))
idx = 122

grid_data = d.iloc[idx].to_numpy().reshape(28,28)
plt.imshow(grid_data, interpolation = "none", cmap = "gray")
plt.show()

print("the number given in the lable is",l[idx])

"""#1.1 Write a piece of code to run the PCA algorithm on this data-set. Show the images of the principal components that you obtain. How much of the variance in the data-set is explained by each of the principal components?"""

def changeDataStructure(d):
    data_numpy=d.to_numpy()
    data_numpy=data_numpy.transpose()
    return data_numpy
def dataCenter(d):
    mean=d.mean(axis=1,keepdims=True)
    d=d - mean
    return d        
d = changeDataStructure(d)
d = dataCenter(d)
print(d.shape)

def findCovariance(d):
    cov=d@d.transpose()
    return cov
cov = findCovariance(d)
def findEigen(cov):
    eigen_value,eigen_vector= np.linalg.eig(cov)
    return eigen_value,eigen_vector
eigen_value,eigen_vector = findEigen(cov)
highest_eigen_val_index = eigen_value.argsort()[::-1]
def findTopKEigenVec(eigen_vector,eigen_value,k):
    n=eigen_value.shape[0]
    #highest_eigen_val_index = eigen_value.argsort()[::-1]
    w=[] #top k eigen vectors are stored
    i = 0
    while i < k:
        w.append(np.array(eigen_vector[:,highest_eigen_val_index[i]]))
        w[i]=w[i].reshape(eigen_vector.shape[1],1)
        i+=1
    return w

k=5


w=findTopKEigenVec(eigen_vector,eigen_value,k)
print(w[0].shape)

def showimg(new_data):
  df2 = pd.DataFrame(new_data)
  plt.figure(figsize=(5,5))
  grid_data = df2.to_numpy().reshape(28,28)
  plt.imshow(grid_data, interpolation = "none", cmap = "gray")
  plt.show()

i = 0
while i < k: 
  showimg(w[i])
  i += 1

#Variance Calculation
s = sum(eigen_value)
j = 0
while j < k:
  print("variance by PC[" ,  highest_eigen_val_index[j] ,"] is " , eigen_value[highest_eigen_val_index[j]])
  print("percentage variance by PC[" ,  highest_eigen_val_index[j] ,"] is " , ((eigen_value[highest_eigen_val_index[j]])/s)*100)
  j = j+1

"""#1.2 Reconstruct the dataset using different dimensional representations. How do these look like? If you had to pick a dimension d that can be used for a downstream task where you need to classify the digits correctly, what would you pick and why?

"""

showimg(d[:,0:1]) #test image
r_list = [25,50,100,200,500]
for r in r_list:
  w=findTopKEigenVec(eigen_vector,eigen_value,r)
  w = np.array(w)
  w= w[:,:,0]
  w=w.transpose()
  d2 = d.transpose()
  new_data = (d2[0:1,] @ w) @ w.transpose()
  showimg(new_data)

"""# 1.3 Write a piece of code to implement the Kernel PCA algorithm on this dataset.Plot the projection of each point in the dataset onto the top-2 components for each kernel. Use one plot for each kernel and in the case of (B), use a different plot for each value of σ."""

#Slicing the data because its to large dataset
d = d[:,0:1000] 
def plot(x,y):
    plt.figure(figsize=(10, 10))
    plt.title(title,fontsize=20)
    plt.scatter(x,y,label="Data",c = "red")
    plt.xlabel('new X', fontsize=15)
    plt.ylabel('new Y', fontsize=15)
    plt.legend(loc="upper left",fontsize=10)
    plt.show()

def polynomial_kernel(x,y,p):
    value=(x.transpose() @ y) + 1
    value=value**p;
    return value;
def Eigenvector_Normalization(eigenValue,beta,k): 
    
    sorted_eigen=np.sort(eigenValue)[::-1]
    i = 0
    while i < k:
        beta[i]=beta[i]/(sorted_eigen[i] ** 0.5)
        i+=1
    return beta

def gaussian_kernel(x,y,sigma):
    sub_val=x-y
    power_term=(-(sub_val.transpose() @ sub_val))/(2*(sigma**2))
    value=np.exp(power_term)
    return value

def computePolynomilaKernelMatrix(d,polynomial):
    size=d.shape[1]
    K=np.zeros([size,size])
    i = 0
    while i < size:
        j = 0
        while j < size:
            K[i][j]=polynomial_kernel(d[:,i],d[:,j],polynomial)
            j +=1
        i+=1
    
    return K

def computeGaussianKernelMatrix(d,sigma):
    size=d.shape[1]
    K=np.zeros([size,size])
    i = 0
    while i < size:
        j = 0
        while j < size:
            K[i][j]=gaussian_kernel(d[:,i],d[:,j],sigma)
            j = j+1
        i = i + 1
    
    return K
    
def Kernel_Centring(K):
    sp_val = np.full((K.shape[0],K.shape[0]), 1/K.shape[0])
    K = K - np.dot(sp_val,K) - np.dot(K,sp_val) + sp_val @ K @ sp_val
    return K
def KPCA(K,title):
    K=Kernel_Centring(K)
    K_eigenvalue,K_eigenvector=findEigen(K)
    beta=findTopKEigenVec(K_eigenvector,K_eigenvalue,2)
    alpha=Eigenvector_Normalization(K_eigenvalue,beta,2)
    Phy_of_x = K @ alpha[0]
    Phy_of_y = K @ alpha[1]
    plot(Phy_of_x,Phy_of_y)

diff_p=[2,3,4]
for p in diff_p:
    K=computePolynomilaKernelMatrix(d,p)
    title="Polynomial Kernel of p = "+str(p)
    KPCA(K,title)
sigmas=np.arange(0.1,1.1,0.1)
for sigma in sigmas:
    K=computeGaussianKernelMatrix(d,sigma)
    title="Gaussian Kernel of sigma = "+str(sigma)
    KPCA(K,title)