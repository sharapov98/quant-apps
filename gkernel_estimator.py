import numpy as np
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt


class GKernelEstimator(BaseEstimator):
    """"""

    def __init__(self, bandwidth=0.01):
        self.bandwidth = bandwidth
        self.y_training = None
        self.x_training = None

    def fit(self, x, y):
        """
        Fit the estimator
        Args:
            x: x_training data
            y: y_training data

        Returns: Self, trained

        """
        self.x_training = x
        self.y_training = y
        return self

    def _generate_kernel_array(self, new_x, bandwidth):
        """
        Generate an array of the kernel outputs given new input x
        Args:
            new_x: bandwidth of the gaussian
            bandwidth: scalar of new point

        Returns: 1D array with the kernel outputs

        """
        diff = new_x - self.x_training
        out = np.exp(- ((diff ** 2) / bandwidth))
        return out

    def _calculate_prediction(self, new_x, bandwidth):
        """
        calculate prediction given new x value
        Args:
            new_x: a scalar of new x
            bandwidth: bandwidth of gaussian

        Returns: scalar of prediction

        """
        kernel_array = self._generate_kernel_array(new_x, bandwidth)
        total_added_weight = np.sum(kernel_array)

        pred = (kernel_array @ self.y_training) / total_added_weight
        return pred

    def predict(self, x_values):
        """

        Args:
            x_values: an array of x-values to predict

        Returns: an array of predicted y-values

        """
        pred_many_x = np.vectorize(self._calculate_prediction)
        pred_res = pred_many_x(x_values, self.bandwidth)
        return pred_res

    def plot(self, resolution):
        """
        Plots the fitted function with training data as scatter plot on top
        Args:
            resolution: Number of points to plot the function

        Returns: None

        """
        x_min = np.amin(self.x_training)
        x_max = np.amax(self.x_training)
        x_for_plot = np.linspace(start=x_min, stop=x_max, num=resolution)
        y_for_plot = self.predict(x_for_plot)
        plt.plot(x_for_plot, y_for_plot)
        plt.scatter(self.x_training, self.y_training, c='red')
