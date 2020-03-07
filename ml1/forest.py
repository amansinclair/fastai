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
        branch_value = None
        n_features = X.shape[1]
        n_samples = X.shape[0]
        for feature in range(n_features):
            idx = np.argsort(X[:, feature])
            yf = y[idx]
            feature_lowest_error = np.Inf
            feature_lowest_branch = None
            for i in range(1, n_samples):
                value_low = np.mean(yf[:i])
                value_high = np.mean(yf[i:])
                error_low = np.sum((yf[:i] - value_low) ** 2)
                error_high = np.sum((yf[i:] - value_high) ** 2)
                error = error_low + error_high
                if error < feature_lowest_error:
                    feature_lowest_error = error
                    feature_lowest_branch = i
            if feature_lowest_error < lowest_error:
                best_feature = feature
                lowest_error = feature_lowest_error
                branch_value = (
                    X[feature_lowest_branch, feature]
                    + X[feature_lowest_branch - 1, feature]
                ) / 2
        return best_feature, branch_value, lowest_error


if __name__ == "__main__":
    print("-" * 30)
    a = np.random.rand(10, 5)
    dt = DecisionTree()
    y = np.random.rand(10)
    print(a)
    print(y)
    print("-" * 30)
    print(dt.create_branch(a, y))
    estimator = DTR(max_depth=1)
    estimator.fit(a, y)
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has %s nodes and has "
        "the following tree structure:" % n_nodes
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print(
                "%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                "node %s."
                % (
                    node_depth[i] * "\t",
                    i,
                    children_left[i],
                    feature[i],
                    threshold[i],
                    children_right[i],
                )
            )
    print()

