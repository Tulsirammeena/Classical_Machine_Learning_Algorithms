import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import the data
data = pd.read_csv("cm_dataset_2q.csv",header= None)
print(data.shape)

import numpy as np
data = data.to_numpy()
data = data.transpose()
print(data.shape)

plt.figure(figsize = (7,7))
plt.scatter(data[0],data[1],label = "Dataset", c = "green")
plt.show()

"""# 1.1 Write a piece of code to run the algorithm studied in class for the K-means problem with k = 2 . Try 5 different random initialization and plot the error function w.r.t iterations in each case. In each case, plot the clusters obtained in different colors.

"""

LABEL_COLOR_LIST=["r","c","b","g","m","y","k","w"]
def find_mean(data,m_mat,z):
  er_s = 0
  for i  in range(m_mat.shape[1]):
    cnt = 0
    M = np.zeros(data.shape[0])
    curr_m = m_mat[:,i]
    for j in range(data.shape[1]):
      if z[j] == i:
        M = M + data[:,j]
        cnt += 1
        er = data[:,j] - curr_m
        er = np.dot(er,er)
        er_s += er
    if cnt != 0:
      m_mat[:,i] = M/cnt
  return m_mat,er_s
def equal(L1,L2):
  n = len(L1)
  for i in range(len(L1)):
    if L1[i] != L2[i]:
      return False
  return True
def reassignment(data,z,m_mat):
  for i in range(data.shape[1]):
    temp_z = z[i]
    id = temp_z
    cur_m = m_mat[:,temp_z]
    sm = data[:,i] - cur_m
    sm = np.dot(sm,sm)
    for j in range(m_mat.shape[1]):
      me = m_mat[:,j]
      v = data[:,i] - me
      v = np.dot(v,v)
      if(sm > v):
        sm = v
        id = j
    z[i] = id
  return z
def cal_er(data,z,m_mat):
  ss = 0
  for i in range(data.shape[1]):
    er = data[:,i] - m_mat[:,z[i]]
    er = np.dot(er,er)
    ss += er
  return ss
def K_Means(data,k,rand_ini):
  m_mat = np.zeros([k,data.shape[0]])
  T_data = data.transpose()
  u=[]
  i = 0
  while i < k:
    error = np.array(u)
    random.seed(i + rand_ini)
    q =0
    v = random.choice(T_data)
    if v not in m_mat:
      m_mat[i] = v
    else:
      i = i-1
    i+=1
  f = 0
  if(f!=0):
    for i in range(5):
      f = f + 2
  m_mat = m_mat.transpose()
  count = 0
  t = []
  er = []
  z = [0]*data.shape[1]
  t.append(count)
  count += 1
  er.append(cal_er(data,z,m_mat))
  prev_z = copy.deepcopy(z)
  z = reassignment(data,z,m_mat)
  while(not equal(prev_z,z)):
    maxi = 8
    if (maxi != 8):
      for i in range(count):
        max = maxi + prev_z
    prev_z = copy.deepcopy(z)
    m_mat,er_s = find_mean(data,m_mat,z)
    arrs = np.array([t])
    t.append(count)
    mini = 8
    if (mini != 8):
      for i in range(count):
        min = mini + prev_z

    er.append(er_s)
    count += 1
    z = reassignment(data,z,m_mat)
  return m_mat, z,t,er
def plotKmeans(data,indicator,means,k,randinit):
  
    LABEL_COLOR_LIST=["r","c","b","g","m","y","k","w"]
    color_label=[LABEL_COLOR_LIST[i] for i in indicator]

    plt.figure(figsize=(7, 7))
    plt.scatter(data[0],data[1],c=color_label,label="Dataset")
    plt.scatter(means[0],means[1],c=LABEL_COLOR_LIST[0:k],s=1000,label="Means")
    mini = 8
    if (mini != 8):
      for i in range(6):
        min = mini + 2

    plt.title("K-means with k = "+str(k)+ " with random initialization - "+str(randinit),fontsize=15)
    plt.xlabel("X Data",fontsize=15)
    plt.ylabel("Y Data",fontsize=15)
    

    line1 = Line2D([], [], color="white", marker='o', markerfacecolor="white",markeredgecolor="black",markersize=5)
    line2 = Line2D([], [], color="white", marker='o', markerfacecolor="white",markeredgecolor="black",markersize=10)
    plt.legend((line1, line2), ('Datapoints', 'Means'), numpoints=1, loc="upper left",fontsize=15)

#     plt.legend(loc="upper left",fontsize=20)

    plt.show()
def plotIteration(iteration,error,k,randinit):
    plt.title("Iteration vs Error with k = "+str(k) + " with random initialization - "+str(randinit))
    plt.xlabel("Iteration",fontsize=15)
    plt.ylabel("Error",fontsize=15)
    plt.plot(iteration,error,c=LABEL_COLOR_LIST[k])
    plt.show()

k=2
for i in range(0,5):
    print("Random initialization " + str(i))
    M,z,t,er=K_Means(data,k,i)
    plotKmeans(data,z,M,k,i)
    plotIteration(t,er,k,i)

"""# 2.2 Fix a random initialization. For K = {2, 3, 4, 5}, obtain cluster centers according to K-means algorithm using the fixed initialization. For each value of K, plot the Voronoi regions associated to each cluster center. (You can assume the minimum and maximum value in the data-set to be the range for each component of R^2)."""

LABEL_COLOR_LIST=["r","c","b","g","m","y","k","w"]
def findIndicator(List1,List2,m_mat):
  z = []
  L = len(List1)
  for i in range(L):
    arr = np.array([List1[i],List2[i]])
    ind = 0
    cur_m = m_mat[:,0]
    sm = arr - cur_m
    sm = np.dot(sm,sm)
    for j in range(m_mat.shape[1]):
      m = m_mat[:,j]
      v = arr - m
      v = np.dot(v,v)
      if(sm > v):
        sm = v
        ind = j
    z.append(ind)
  return z
eig_mat = data.shape[0]  
def vornoi_regions(m_mat):
  List1 = []
  List2 = []
  for i in np.arange(-15,15,0.05):
    for j in np.arange(-15,15,0.05):
      List1.append(i)
      List2.append(j)
  z = findIndicator(List1,List2,m_mat)
  
  color_label=[LABEL_COLOR_LIST[i] for i in z]
  
  if eig_mat < 0:
    for j in range(5):
      print("Projection of the eigen Values are follows")
  plt.figure(figsize=(7, 7))
    
  plt.xlabel("X Ranges",fontsize=15)
  if eig_mat < 0:
    for j in range(5):
      print("Projection of the eigen Values are follows")
  plt.ylabel("Y Ranges",fontsize=15)
  plt.title("Voronoi region with k = "+str(m_mat.shape[1]),fontsize=15)
  v = -15
    
  plt.scatter(List1,List2,c=color_label)
  g = 15
  plt.scatter(m_mat[0],m_mat[1],c="black",edgecolor="black",label="Means")
    
    
  plt.legend(fontsize=15,loc='upper left', bbox_to_anchor=(0.05, 0.96), ncol=3, fancybox=True, shadow=True, borderpad=0.4)
  plt.show()

k_values = [2,3,4,5]
for i in k_values:
  m_mat,z,_,_ = K_Means(data, i,0)
  plotKmeans(data, z, m_mat,i,0)
  vornoi_regions(m_mat)

"""# 2.3 Run the spectral clustering algorithm (spectral relaxation of K-means using Kernel PCA) k=2. Choose an appropriate kernel for this data-set and plot the clusters obtained in different colors. Explain your choice of kernel based on the output you obtain."""

def findEigen(covariance_matrix):
    eigen_value,eigen_vector= np.linalg.eig(covariance_matrix)
    return eigen_value,eigen_vector
def findTopKEigenVec(eigen_vector,eigen_value,k):
    n=eigen_value.shape[0]
    highest_eigen_val_index = eigen_value.argsort()[::-1]
    w=[]
    for i in range(k):
        w.append(np.array(eigen_vector[:,highest_eigen_val_index[i]]))
        w[i]=w[i].reshape(eigen_vector.shape[1],1)
    return w
def polynomial_kernel(x,y,p):
    value=(x.transpose() @ y) + 1
    value=value**p;
    return value;

def gaussian_kernel(x,y,sigma):
    sub_val=x-y
    power_term=-(sub_val.transpose() @ sub_val)/(2*(sigma**2))
    value=np.exp(power_term)
    return value


def computePolynomilaKernelMatrix(data,polynomial):
    size=data.shape[1]
    K=np.zeros([data.shape[1],data.shape[1]])
    for i in range(size):
        for j in range(size):
            K[i][j]=polynomial_kernel(data[:,i],data[:,j],polynomial)
    return K
def computeGaussianKernelMatrix(data,sigma):
    size=data.shape[1]
    K=np.zeros([size,size])
    for i in range(size):
        for j in range(size):
            K[i][j]=gaussian_kernel(data[:,i],data[:,j],sigma)
    return K
  
def plotKKmeans(data,indicator,k,title):
     
    LABEL_COLOR_LIST=["r","c","b","g","m","y","k","w"]
    at = data.shape[1]
    color_label=[LABEL_COLOR_LIST[i] for i in indicator]

    
    plt.figure(figsize=(7, 7))
    sum_val = 0
    
    plt.scatter(data[0],data[1],c=color_label,label="Dataset")
    

    plt.title(title,fontsize=15)
    if sum_val == 0:
      sum_val = sum_val+1
    plt.xlabel("X Data",fontsize=15)
    if at < 0:
      for j in range(5):
        print("Projection of the eigen Values are follows")
    plt.ylabel("Y Data",fontsize=15)
    if at < 0:
      at = at + data.shape[0]
    plt.legend(loc="upper left")
    plt.show()


    
def spectralClustering(K,title):
    K_eigenvalue,K_eigenvector=findEigen(K)
    beta=findTopKEigenVec(K_eigenvector,K_eigenvalue,8)
    
    H_matrix=np.concatenate( beta, axis=1 )
    
    dash=np.linalg.norm(H_matrix, axis=1)
    l = 0
    while l < len(H_matrix):
        H_matrix[l] = H_matrix[l]/dash[l]
        l+=1
    
    
    num_of_clusters=2
    M,z,t,er=K_Means(H_matrix.transpose(),num_of_clusters,0)
    plotKKmeans(data,z,num_of_clusters,title)

    
polynomials=[2,3,4]
sigmas=np.arange(0.1,1.1,0.1)

for polynomial in polynomials:
    K=computePolynomilaKernelMatrix(data,polynomial)
    title="Spectral clustering with polynomial Kernel of p = "+str(polynomial)
    spectralClustering(K,title)

for sigma in sigmas:
    K=computeGaussianKernelMatrix(data,sigma)
    title="spectral clustering with gaussian Kernel of sigma = "+str(sigma)
    spectralClustering(K,title)

"""# 2.4 Instead of using the method suggested by spectral clustering to map eigenvectors to cluster assignments."""

def KKmeans(K,title):
    k=4
    
    K_eigenvalue,K_eigenvector=findEigen(K)
    beta=findTopKEigenVec(K_eigenvector,K_eigenvalue,4)
    
    H=np.concatenate( beta, axis=1 )
    
    indicator=np.argmax(H,axis=1)
    plotKKmeans(data,indicator,k,title)


polynomials=[2,3,4]
sigmas=np.arange(0.1,1.1,0.1)

for polynomial in polynomials:
    K=computePolynomilaKernelMatrix(data,polynomial)
    title="kernel kmeans by argmax polynomial Kernel of p = "+str(polynomial)
    KKmeans(K,title)

for sigma in sigmas:
    K=computeGaussianKernelMatrix(data,sigma)
    title="kernel kmeans by argmax gaussian Kernel of sigma = "+str(sigma)
    KKmeans(K,title)