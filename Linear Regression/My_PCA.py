import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
class My_PCA():

    def getStandardizedData(self,my_data):
        standardized_my_data = (my_data - np.mean(my_data, axis=0)) / np.std(my_data, axis=0)
        return standardized_my_data


    def my_variance(x):
        x_mean = np.mean(x)
        # Formula is Xi - XMean/N
        N = len(x)
        sum = 0
        for each in x:
            sum += math.pow((each - x_mean), 2)

        return sum / N

    def my_covariance(self,my_data):
        return np.cov(my_data,rowvar=False)

    def performPCA(self,data,standardizeLogic):
        #1. Mean Centre the Data
        if standardizeLogic:
            data = data - np.mean(data, axis=0)
        #Else do not Mean centre

        #2. Covariance Matrix
        myCOV = self.my_covariance(data)
        eigenValues, eigenVectors = LA.eig(myCOV)
        idx = np.argsort(eigenValues)[::-1]
        eigenVectors = eigenVectors[:,idx]
        eigenValues = eigenValues[idx]
        projections = np.dot(data,eigenVectors)
        print projections
        #fig = plt.figure()
        # ax = fig.add_subplot(221, projection='3d')
        # ax.set_title('Original')
        # x = data[:,0]
        # y = data[:,1]
        # z = data[:,2]
        # ax.scatter(x, y, z, c='r', marker='o')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        #ax2 = fig.add_subplot(224)
        #plt.title('Plotting First 2 Principal Components')

        firstPC1X_20 = projections[:20, 0]
        firstPC1Y_20 = projections[:20, 1]

        firstPC2X_20 = projections[21:, 0]
        firstPC2Y_20 = projections[21:, 1]

        #plt.scatter(firstPC1X_20, firstPC1Y_20, c='r')
        #plt.scatter(firstPC2X_20, firstPC2Y_20, c='b')

        #plt.xlabel('Principal Component 1')
        #plt.ylabel('Principal Component 2')
        #plt.show()
        return eigenValues,eigenVectors,projections

    def plotScree(self,eigenValues):
        # ----------- Plotting Variance plot -----------
        varianceExplained = []
        eigV = []
        cnt = 1
        print 'Printing EigenValues'
        print eigenValues
        for num in eigenValues:
            varianceExplained.append((num / np.sum(eigenValues)) * 100)
            eigV.append(cnt)
            cnt += 1

        import matplotlib.pyplot as plt
        plt.title('Scree Plot')
        plt.xlabel('Principal Components---->')
        plt.ylabel('Variance Explained')
        plt.xticks(eigV)
        plt.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
        plt.scatter(eigV, varianceExplained, c='r', marker='o')
        #plt.show()



            # my_data = np.genfromtxt('dataset_1.csv', delimiter=',', skip_header=1)
# pca = My_PCA()
# pca.performPCA(data=my_data)