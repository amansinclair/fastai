import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn import tree


class DecisionTree:
    def __init__(self, max_depth=0):
        self.max_depth = max_depth

    def fit(self, X, y):
        pass

    def create_branch(self, X, y):
        value = np.mean(y)
        error = np.sum((y - value) ** 2)
        mse = error ** 0.5
        best_feature = None
        lowest_error = np.Inf
        best_break_point = None
        n_features = X.shape[1]
        for feature in range(n_features):
            Xf = X[:, feature].copy()
            break_points = self.create_break_points(Xf)
            feature_lowest_error = np.Inf
            feature_lowest_break_point = None
            for break_point in break_points:
                low = y[np.where(Xf <= break_point)]
                high = y[np.where(Xf > break_point)]
                error_low = np.sum((low - np.mean(low)) ** 2)
                error_high = np.sum((high - np.mean(high)) ** 2)
                error = error_low + error_high
                if error < feature_lowest_error:
                    feature_lowest_error = error
                    feature_lowest_break_point = break_point
            if feature_lowest_error < lowest_error:
                best_feature = feature
                lowest_error = feature_lowest_error
                best_break_point = feature_lowest_break_point
        return best_feature, best_break_point, lowest_error

    def create_break_points(self, X):
        unique = np.unique(X)
        break_points = np.zeros(len(unique) - 1)
        for i in range(len(unique) - 1):
            break_points[i] = (unique[i] + unique[i + 1]) / 2
        return break_points
