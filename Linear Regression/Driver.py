''' Program for Homework 1: Machine Learning
    Author: Kshitij Gorde
    ID: 800966672
'''
from My_PCA import My_PCA
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt
fig = plt.figure()
def question_1_plots(input_data,principal_component):

    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Plot')
    ax1.scatter(input_data['x'], input_data['y'], color='red', label='x v/s y')
    ax1.plot(input_data['x'], input_data['y_theoretical'], color='blue', label='x v/s y_theoretical')
    ax1.plot([0, principal_component[0, 0]], [0, principal_component[1, 0]],
               color='green', linewidth=4, label='PC1 Axis')
    ax1.legend()
    plt.show()


def plot_regression_line(estimated_y,input_data):
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Plotting the Regression Line')
    ax1.scatter(input_data['x'],input_data['y'],color='red')
    ax1.plot(input_data['x'], estimated_y, color='purple')
    plt.show()

def plot_all_together(input_data,principal_component,estimated_y):
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Plot')
    ax1.scatter(input_data['x'], input_data['y'], color='red', label='x v/s y')
    ax1.plot(input_data['x'], input_data['y_theoretical'], color='blue', label='x v/s y_theoretical')
    ax1.plot([0, principal_component[0, 0]], [0, principal_component[1, 0]],
             color='green', linewidth=4, label='PC1 Axis')
    ax1.plot(input_data['x'], estimated_y, color='purple', label='Regression Line')
    ax1.legend()
    plt.show()


def plot_diabetes_regression(y_predicted,testing_dataset):
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Diabetes Dataset Regression Analysis')
    ax1.scatter(testing_dataset[:, 0], testing_dataset[:, 1], color='red', label='testing x v/s testing y')
    ax1.plot(testing_dataset[:, 0], y_predicted, color='black', label='testing x v/s predicted y')
    ax1.legend()
    plt.show()



def main():
    # Load the dataset
    df = pd.read_csv('linear_regression_test_data.csv')
    # Load x and y only
    input_data = df[['x','y']]
    pca_object = My_PCA()
    eigen_values, eigen_vectors, projections = pca_object.performPCA(input_data,False)
    print 'Eigen Values are: '
    print eigen_values
    print 'Eigen Vectors are: '
    print eigen_vectors
    print 'Projections are: '
    print projections

    #question_1_plots(df,eigen_vectors)
    linear_regression = LinearRegression()

    estimated_y = linear_regression.linear_regression(input_data)
    #plot_regression_line(estimated_y,input_data)
    # Now plot all together

    #plot_all_together(df,eigen_vectors,estimated_y)


    # --------- Question 2------------
    # Linear Regression using scikit-learn
    y_predicted,testing_dataset,training_dataset = linear_regression.linear_regression_sklearn() #Diabetes dataset

    plot_diabetes_regression(y_predicted,testing_dataset)


if __name__ == '__main__':
    main()