import numpy as np
import sys
import plotly
import plotly.plotly as py
import plotly.figure_factory as ff
import math
import matplotlib.pyplot as plt
import pylab
plotly.tools.set_credentials_file(username='kshitijgorde', api_key='4nnHJACIYBs4nVin7YCa')



def plotClusters(data, clusterDictionary,threshold):
    import colorsys
    N = len(clusterDictionary)
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    clusters = clusterDictionary[threshold]
    uniqueColorsRequired = set(clusters)
    colorsDictionary = {}
    for everyItem in uniqueColorsRequired:
        colorsDictionary[everyItem] = everyItem%23

    for i in xrange(len(data)):
        plt.scatter(data[i][0],data[i][3],color = RGB_tuples[clusters[i]])

    plt.show()
    return

def singleLinkageAgglomerative(distanceMatrix):
    clusterDict = {}
    clusterNo = 0
    first_clusters = []
    for i in xrange(0, len(distanceMatrix)):
        first_clusters.append(i)

    clusters = list(first_clusters)

    threshold = 0
    clusterDict[threshold] = first_clusters
    rows, cols = distanceMatrix.shape
    minimum = sys.maxint
    temp = distanceMatrix.shape[0]
    while temp > 1:
        rows, cols = distanceMatrix.shape
        minimum = sys.maxint
        print 'New Rows, Cols'
        print rows, cols
        for i in range(rows):
            for j in range(cols):
                if i != j:  # Not diagonal
                    if distanceMatrix[i][j] <= minimum:
                        minimum = distanceMatrix[i][j]
                        minRow = i
                        minCol = j
        print 'Here....New minimum, minRow, minCol'
        print minimum
        print minRow
        print minCol
        # Here I have to update
        clusterNo = min(clusters[minRow], clusters[minCol])
        tobeUpdated = max(clusters[minRow], clusters[minCol])
        clusters[minRow] = clusterNo
        clusters[minCol] = clusterNo
        # Where ever there is tobeUpdated in clusters, update those to be clusterNo
        for i in xrange(len(clusters)):
            if clusters[i] == tobeUpdated:
                clusters[i] = clusterNo
        threshold += 1

        updatedClusters = list(clusters)
        clusterDict[threshold] = updatedClusters
        # Now the tricky Part..From point (minRow, minCol) find distances to all other points
        # Update the Distance Matrix
        count = 0
        for k in range(rows - 1):

            if k != minRow and k!= minCol:
                distanceMatrix[minCol][count] = min(distanceMatrix[minCol][count], distanceMatrix[minRow][count])
                distanceMatrix[count][minCol] = distanceMatrix[minCol][count]
            count += 1
        distanceMatrix[:, minRow] = sys.maxint
        distanceMatrix[minRow] = sys.maxint
        print 'New shape'
        print distanceMatrix.shape
        temp -= 1
        print distanceMatrix
        print 'Single Linkage Clustering Completed....'

    print clusterDict
    return clusterDict



def completeLinkageAgglomerative(distanceMatrix):
    clusterDict = {}
    clusterNo = 0
    first_clusters = []
    for i in xrange(0, len(distanceMatrix)):
        first_clusters.append(i)

    clusters = list(first_clusters)

    threshold = 0
    clusterDict[threshold] = first_clusters
    rows, cols = distanceMatrix.shape
    minimum = sys.maxint
    temp = distanceMatrix.shape[0]
    while temp > 1:
        rows, cols = distanceMatrix.shape
        minimum = sys.maxint
        print 'New Rows, Cols'
        print rows, cols
        for i in range(rows):
            for j in range(cols):
                if i != j:  # Not diagonal
                    if distanceMatrix[i][j] <= minimum and distanceMatrix[i][j]!=-1:
                        minimum = distanceMatrix[i][j]
                        minRow = i
                        minCol = j
        print 'Here....minimum, minRow, minCol'
        print minimum
        print minRow
        print minCol
        # Here I have to update
        clusterNo = min(clusters[minRow], clusters[minCol])
        tobeUpdated = max(clusters[minRow], clusters[minCol])
        clusters[minRow] = clusterNo
        clusters[minCol] = clusterNo
        # Where ever there is tobeUpdated in clusters, update those to be clusterNo
        for i in xrange(len(clusters)):
            if clusters[i] == tobeUpdated:
                clusters[i] = clusterNo
        threshold += 1

        updatedClusters = list(clusters)
        clusterDict[threshold] = updatedClusters
        # Now the tricky Part..From point (minRow, minCol) find distances to all other points
        # Update the Distance Matrix
        count = 0
        for k in range(temp):

            if k != minRow and k!= minCol:
                print 'Now Ill be taking max from :',str(distanceMatrix[minCol][count])+' and '+str(distanceMatrix[minRow][count])
                distanceMatrix[minCol][count] = max(distanceMatrix[minCol][count], distanceMatrix[minRow][count])
                print distanceMatrix[minCol][count]
                distanceMatrix[count][minCol] = distanceMatrix[minCol][count]
            count += 1
        distanceMatrix[:, minRow] = -1
        distanceMatrix[minRow] = -1
        print 'New shape'
        print distanceMatrix.shape
        temp -= 1
        print distanceMatrix
    print 'Complete Linkage Clustering Completed...'
    print clusterDict
    return clusterDict



def averageLinkageAgglomerative(distanceMatrix):
    clusterDict = {}
    clusterNo = 0
    first_clusters = []
    for i in xrange(0, len(distanceMatrix)):
        first_clusters.append(i)

    clusters = list(first_clusters)

    threshold = 0
    clusterDict[threshold] = first_clusters
    rows, cols = distanceMatrix.shape
    minimum = sys.maxint
    temp = distanceMatrix.shape[0]
    while temp > 1:
        rows, cols = distanceMatrix.shape
        minimum = sys.maxint
        print 'New Rows, Cols'
        print rows, cols
        for i in range(rows):
            for j in range(cols):
                if i != j:  # Not diagonal
                    if distanceMatrix[i][j] <= minimum and distanceMatrix[i][j] != -1:
                        minimum = distanceMatrix[i][j]
                        minRow = i
                        minCol = j
        print 'Here....minimum, minRow, minCol'
        print minimum
        print minRow
        print minCol
        # Here I have to update
        clusterNo = min(clusters[minRow], clusters[minCol])
        tobeUpdated = max(clusters[minRow], clusters[minCol])
        clusters[minRow] = clusterNo
        clusters[minCol] = clusterNo
        # Where ever there is tobeUpdated in clusters, update those to be clusterNo
        for i in xrange(len(clusters)):
            if clusters[i] == tobeUpdated:
                clusters[i] = clusterNo
        threshold += 1

        updatedClusters = list(clusters)
        clusterDict[threshold] = updatedClusters
        # Now the tricky Part..From point (minRow, minCol) find distances to all other points
        # Update the Distance Matrix
        count = 0
        for k in range(rows - 1):

            if k != minRow and k!=minCol:
                print 'Now Ill be taking Average of :', str(distanceMatrix[minCol][count]) + ' and ' + str(
                    distanceMatrix[minRow][count])
                distanceMatrix[minCol][count] = (distanceMatrix[minCol][count] + distanceMatrix[minRow][count]) / 2
                print distanceMatrix[minCol][count]
                distanceMatrix[count][minCol] = distanceMatrix[minCol][count]
            count += 1
        distanceMatrix[:, minRow] = -1
        distanceMatrix[minRow] = -1
        print 'New shape'
        print distanceMatrix.shape
        temp -= 1
        print distanceMatrix
    print 'Average Linkage Clustering Completed...'
    print clusterDict
    return clusterDict









def findMinimumIndices(distanceMatrix,technique):
    if technique == 'Single Linkage':
        clusterDictionary = singleLinkageAgglomerative(distanceMatrix)
    elif technique == 'Complete Linkage':
        clusterDictionary = completeLinkageAgglomerative(distanceMatrix)
    elif technique == 'Average Linkage':
        clusterDictionary= averageLinkageAgglomerative(distanceMatrix)
    else:
        print '\nTechnique not Recognized'
        exit(1)

    return clusterDictionary




# data = np.array([[0.40,0.53],[0.22,0.38],[0.35,0.32],[0.26,0.19],[0.08,0.41],[0.45,0.30]])
# print(data)

from sklearn.metrics.pairwise import pairwise_distances

# import matplotlib.pyplot as plt
# plt.scatter(data[:,0],data[:,1])
# plt.show()
from sklearn.datasets import load_iris
data = load_iris().data

print load_iris().feature_names
initialDistances = pairwise_distances(data,metric='euclidean')
print(initialDistances)
clusterDictionary = findMinimumIndices(initialDistances,technique='Average Linkage')

plotClusters(data,clusterDictionary,146)

