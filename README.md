# classical-machine-learning-from-scratch
It's always amusing how machine learning works. Most of us use well-known frameworks such as scikit-learn. But what if we want to know the real working of some algorithms?. Here, I've developed my own small repository of a few machine learning algorithms using only Numpy and Pandas

# Here are a few algorithms that I've developed from scratch

# 1. Principal Component Analysis
We're all aware of the 'Curse of High Dimensionality' when dealing with most machine learning algorithms. Principal Component Analysis to the rescue!
Fun-fact, PCA will preserve the total variance in your dataset. 
All you have to do is calculate the eigenvalues and eigenvectors and sort them in descending order. Finally, just take projects of your datapoints on these eigenvectors (by checking how much variance you want to preserve in your dataset) and there you have your Principal components.

# 2 . Agglomerative Clustering
When it comes to unsupervised learning, it's always to tough choice to choose a clustering algorithm. Fortunately, agglomerative clustering or heirarchical clustering. This algorithm will cluster all of your datapoints starting from n clusters to just 1. You will thus have a flexibility to define a threshold for your total clusters. I have included, Complete Linkage, Average linkage and Single linkage while calculating cluster linkages. Further, loadings plot and scree plot have also been drawn. Feel free to contribute and add a dendogram functionality.

# 3. K-Means Clustering
K-means is a famous clustering technique in unsupervised learning. However, we know that its main drawback is to know the size of the clusters beforehand. Nevertheless, to develop this algoirthm, all we need is to calculate pariwise_distances and then calculate centroids and repeat the procedure until the algorithm converges.

# 4. Linear Regression
Simple linear regression is a statistical method that allows us to summarize and study relationships between two continuous (quantitative) variables.
All we do here is find that perfect line which fits our data. We start by estimating the coefficients of the equation of the line
y = mx + c. Only here, its y = B0 + B1x

# 5. Linear Discriminant Analysis
We talked about PCA and curse of dimensionality. However, PCA doesn't preserve the discriminating factor in our dataset. That's when LDA comes to the rescue. It is used to find a linear combination of features that characterizes or separates two or more classes of objects or events. The resulting combination may be used as a linear classifier, or, more commonly, for dimensionality reduction before later classification.

# 6. Artificial Neural Network 
Here, we implement our own Neural net with 1 hidden layer. We optimize the cost function using the Gradient Descent Algorithm. Backpropogation is also implemented!. We further extend it for multiple hidden layers


# All above algorithms have been tested against scikit-learn and have shown similar accuracy
