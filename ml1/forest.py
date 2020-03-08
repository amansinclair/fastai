import numpy as np
import pandas as pd
from anytree import NodeMixin, RenderTree


class TreeNode(NodeMixin):
    def __init__(
        self, name, mse, n_samples, value, feature=None, break_point=None, parent=None
    ):
        super().__init__()
        self.name = name
        self.mse = mse
        self.n_samples = n_samples
        self.value = value
        self.feature = feature
        self.break_point = break_point
        self.parent = parent

    def __repr__(self):
        if self.feature:
            fmt = f"Node({self.name}, feature:{self.feature}, breakpoint<=:{self.break_point:.3f}, mse:{self.mse:.3f}, samples:{self.n_samples}, value:{self.value:.3f}"
        else:
            fmt = f"Node({self.name}, mse:{self.mse:.3f}, samples:{self.n_samples}, value:{self.value:.3f}"
        return fmt


class DecisionTree:
    def __init__(self, max_depth=np.Inf):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        depth = 0
        nodes = [None]
        splits = [None]
        is_finished = False
        while depth <= self.max_depth and not is_finished:
            new_nodes = []
            new_splits = []
            node_idx = 0
            is_finished = True
            for parent, split in zip(nodes, splits):
                if not parent:
                    self.root = self.create_node(X, y, "(0,0)", parent)
                    new_nodes.append(self.root)
                    new_splits.append((X, y))
                    is_finished = False
                else:
                    X, y = split
                    for samples, labels in self.split(parent, X, y):
                        if samples.size > 1:
                            is_finished = False
                            name = f"({depth}, {node_idx})"
                            new_nodes.append(
                                self.create_node(samples, labels, name, parent)
                            )
                            new_splits.append((samples, labels))

                            node_idx += 1
            nodes = new_nodes
            splits = new_splits
            depth += 1

    def split(self, node, X, y):
        if node.n_samples > 1:
            break_point = node.break_point
            feature = node.feature
            idx_low = np.where(X[:, feature] <= break_point)
            idx_high = np.where(X[:, feature] > break_point)
            samples_low = X[idx_low]
            samples_high = X[idx_high]
            labels_low = y[idx_low]
            labels_high = y[idx_high]
            return [(samples_low, labels_low), (samples_high, labels_high)]
        else:
            return [(np.array([]), np.array([]))]

    def create_node(self, X, y, name, parent):
        value = np.mean(y)
        error = np.sum((y - value) ** 2)
        mse = error ** 0.5
        best_feature = None
        lowest_error = np.Inf
        best_break_point = None
        n_samples, n_features = X.shape
        for feature in range(n_features):
            Xf = X[:, feature].copy()
            break_points = self.create_break_points(Xf)
            feature_lowest_error = np.Inf
            feature_lowest_break_point = None
            for break_point in break_points:
                error = self.evaluate_break_point(Xf, y, break_point)
                if error < feature_lowest_error:
                    feature_lowest_error = error
                    feature_lowest_break_point = break_point
            if feature_lowest_error < lowest_error:
                best_feature = feature
                lowest_error = feature_lowest_error
                best_break_point = feature_lowest_break_point
        return TreeNode(
            name, mse, n_samples, value, best_feature, best_break_point, parent
        )

    def create_break_points(self, X):
        unique = np.unique(X)
        break_points = np.zeros(len(unique) - 1)
        for i in range(len(unique) - 1):
            break_points[i] = (unique[i] + unique[i + 1]) / 2
        return break_points

    def evaluate_break_point(self, Xf, y, break_point):
        low = y[np.where(Xf <= break_point)]
        high = y[np.where(Xf > break_point)]
        error_low = np.sum((low - np.mean(low)) ** 2)
        error_high = np.sum((high - np.mean(high)) ** 2)
        return error_low + error_high


if __name__ == "__main__":
    X = np.random.rand(100, 20)
    y = np.random.rand(100)
    tree = DecisionTree(max_depth=4)
    tree.fit(X, y)
    print(RenderTree(tree.root))
