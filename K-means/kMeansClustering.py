from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def do_kmeans(dataset, k):
    data = dataset
    centroids = []
    for i in xrange(k):
        randomCluster = np.random.randint(len(dataset),size=1)
        centroids.append(data[randomCluster])

    print 'Random choosing k-centroids..\n'
    print centroids


    #Now calculate pairwise euclidean distances from these centroids
    pairWiseDistances = []
    for i in xrange(len(centroids)):
        pairWiseDistances.append(pairwise_distances(centroids[i],data,metric='euclidean')[0])


    pairWiseDistances = np.array(pairWiseDistances).T
    print 'PairWise Distances'
    print pairWiseDistances
    print 'Min Indexes from Each Row..'
    minIdx = np.argmin(pairWiseDistances,axis=1)
    print minIdx

    while True:
        centroids = np.array([data[np.where(minIdx==idx)].mean(axis=0) for idx in np.unique(minIdx)])
        pairWiseDistances = []
        for i in xrange(len(centroids)):
            pairWiseDistances.append(pairwise_distances(np.array(centroids[i]).reshape(1,len(centroids[i])), data, metric='euclidean')[0])
        pairWiseDistances = np.array(pairWiseDistances).T

        minIdx = np.argmin(pairWiseDistances, axis=1)
        print "Calculating New Centroids and checking for convergence...\n"
        new_centroids = np.array([data[np.where(minIdx == idx)].mean(axis=0) for idx in np.unique(minIdx)])

        if np.array_equal(new_centroids,centroids):
            print 'Convergence Achieved....\n'
            break

    print minIdx
    #---Code for Plotting---------------------------------------------------------------------------------
    finalClusters = np.array([data[np.where(minIdx == idx)] for idx in np.unique(minIdx)])

    #Plt Original
    # figure = plt.figure()
    # ax1 = Axes3D(figure)
    # ax1.scatter(data[:,3],data[:,0],data[:,2])
    # plt.show()


    for i in xrange(len(finalClusters)):
        plt.scatter(finalClusters[i][:,0],finalClusters[i][:,3])

    plt.show()


dataset = load_iris()
data = dataset.data

print data
do_kmeans(data,3)