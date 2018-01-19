import numpy as np
import sklearn.linear_model
from sklearn import datasets

class LinearRegression:

    def linear_regression(self,input_data):
        #B1_hat is cov(x,y)/var(x)

        beta_1_hat = np.cov(input_data['x'], input_data['y'])[0, 1] / np.cov(input_data['x'], input_data['y'])[0, 0]
        # B0_hat = y_mean - (b1 * x_mean)
        beta_0_hat = np.mean(input_data['y']) - beta_1_hat * np.mean(input_data['x'])

        # Estimate y
        y_hat = beta_0_hat + beta_1_hat * input_data['x']

        return y_hat

    def linear_regression_sklearn(self):
        diabetes = datasets.load_diabetes()

        x = diabetes.data[:, 2]
        y = diabetes.target

        input_data = np.column_stack((x, y))

        # Randomly select 20 data points for testing
        testing_locations = np.random.randint(len(x), size=20)

        testing_dataset = input_data[testing_locations, :]
        training_dataset = np.delete(input_data, testing_locations, 0)  # Training data will be whole data - testing data
        temp = len(training_dataset[:, 0])
        # Model Generation
        lm_model = sklearn.linear_model.LinearRegression()
        x_training = training_dataset[:, 0].reshape(temp, 1)
        y_training = training_dataset[:, 1].reshape(temp, 1)
        lm_model.fit(x_training, y_training)

        # Calculate the Predictions
        y_predicted = lm_model.predict(testing_dataset[:, 0].reshape((len(testing_dataset[:, 0]), 1)))

        return y_predicted,testing_dataset,training_dataset
