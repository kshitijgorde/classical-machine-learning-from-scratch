# Kshitij Gorde.
# My own Principal Component Analysis
import math
import numpy as np

np.random.seed(101)
# x = np.random.normal(size=100)
# y = np.random.normal(size=100)
#
# print 'Printing from Numpy...\n'
#
# print '\nCovariance of X: ' + str(np.cov(x))
# print '\nCovariance of X: ' + str(np.cov(y))
# print '\nCovariance of (x,y):\n ' + str(np.cov(x,y))

# Calculate your own
my_data = np.genfromtxt('dataset_1.csv',delimiter=',',skip_header=1)

#Calculate Variance of Variable x in data file

def my_variance(x):
    print 'Variance of x calculated without numpy:'
    x_mean = np.mean(x)
    # Formula is Xi - XMean/N
    N = len(x)
    sum = 0
    for each in x:
        sum += math.pow((each - x_mean), 2)

    return sum / N

def my_covariance(my_data):
    #Mean centre x
    mean_matrix = np.mean(my_data,0)
    X = my_data - mean_matrix
    my_cov = np.dot(np.transpose(X),X) / len(X)
    return my_cov





# ------------- Calculating Q3 with linalg -------------

A = np.array([[0,-1],[2, 3]])

# ---- Printing Eigenvalues and Eigenvectors

eigenVal,eigenVec = np.linalg.eig(A)

print 'EigenValues from Numpy.linalg for Q3.\n'
print eigenVal

print 'EigenVectors from Numpy.linalg for Q3.\n'
print eigenVec





#print my_data
x = my_data[:,0] # Gives only x
y = my_data[:,1]
z = my_data[:,2]


print 'Q1. Calculate the Variance of every variable in the data file.\nAnswer:\n'


print 'Variance of x from the dataset:'
#print '\n'+str(np.cov(x))
X = np.vstack([x,y,z])
print np.cov(X)
#print 'Covariance of dataset with own implementation:\n'
my_cov = my_covariance(my_data)
print my_cov

print 'Variance of x: ' + str(my_cov[0][0])
print 'Variance of y: ' + str(my_cov[1][1])
print 'Variance of z: ' + str(my_cov[2][2])

# ------------- ANSWER ------------------
# Variance of x: 0.080529305884
# Variance of y: 2.09690259152
# Variance of z: 0.080501954879

#----------------------------------------
print '\nQ2. Calculate the Covariance between x & y and y & z.\nAnswer:\n'
print 'cov(x,y) :' + str(my_cov[0][1])
print 'cov(y,z) :' + str(my_cov[1][2])


# ------ Answer ---------------
#  cov(x,y) :0.402026347714
# cov(y,z) :-0.014380262649
# ---------------------------

eigenValues, eigenVectors = np.linalg.eig(my_cov)


mean_matrix = np.mean(my_data,0)
Xmean = my_data - mean_matrix

x = Xmean[:,0]
y = Xmean[:,1]
z = Xmean[:,2]


idx = eigenValues.argsort()[::-1]
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]


print '--------------'
print 'Printing EigenValues\n'
print eigenValues
print 'Printing EigenVectors\n'
print eigenVectors

My_Matrix = np.dot(Xmean,eigenVectors)
print 'Printing My_Matrix'
print My_Matrix

P1 = My_Matrix[:,0]
P2 = My_Matrix[:,1]

# pca = PCA(my_data)
#print 'Printing PCA from sklearn\n'
# from sklearn.decomposition import PCA
# pca=PCA(n_components=2)
# pca.fit(my_data)
# print pca.fit_transform(my_data)


data_r = Xmean * My_Matrix
reduced_data = data_r[:,0:2]



# Reduced data is now 1000 x 2 matrix with reduced dimensions.
# print pca.Y

#------- Plotting --------------

# First plot original scatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt




fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.set_title('Original')
ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


#fig = plt.figure()
ax2 = fig.add_subplot(224)
ax2.set_title('Plotting First 2 Principal Components')
ax2.scatter(reduced_data[:,0],reduced_data[:,1])
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')


plt.show()

# -------------- Please Refer PCA.png for Plots------------------




# Answer------------ Projection Matrix:
# [[ -1.75681915e+00   2.20257289e-03  -8.97784291e-02]
#  [ -8.88177567e-01   1.64657996e-02  -2.54949633e-02]
#  [  5.26487022e-02  -4.46783426e-01   3.83101057e-02]
#  ...,
#  [  3.60428154e-01   3.45422947e-01   4.99235344e-02]
#  [ -2.86808081e+00  -3.85070743e-01   6.02112236e-02]
#  [  1.52319700e+00  -1.45746184e-01  -3.05903854e-02]]

#  ------------- Eigen Values --------------

#   [ 2.17420495  0.08040102  0.00332789]

#  ------------- Eigen Vectors --------------
# [[ 0.18857784  0.00448705  0.982048  ]
#  [ 0.98203351  0.00623651 -0.18860355]
#  [-0.00697082  0.99997049 -0.00323037]]

# --------------------------------------------