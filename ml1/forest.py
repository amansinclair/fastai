import numpy as np
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

    def split(self, sample):
        if not self.is_leaf:
            if sample[self.feature] <= self.break_point:
                return self.children[0]
            else:
                return self.children[1]

    def __repr__(self):
        if self.is_leaf:
            fmt = f"Node({self.name}, mse:{self.mse:.3f}, samples:{self.n_samples}, value:{self.value:.3f}"
        else:
            fmt = f"Node({self.name}, feature:{self.feature}, breakpoint<=:{self.break_point:.3f}, mse:{self.mse:.3f}, samples:{self.n_samples}, value:{self.value:.3f}"
        return fmt


class DecisionTree:
    def __init__(self, max_depth=np.Inf):
        self.max_depth = max_depth
        self.root = None
        self.feature_importance = None

    def __repr__(self):
        if self.root:
            return str(RenderTree(self.root))
        else:
            return "DecisionTree()"

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
        self.feature_importance = self.get_feature_importance(X, y)

    def split(self, node, X, y):
        if node.n_samples > 1 and node.break_point:
            idx_low = np.where(X[:, node.feature] <= node.break_point)
            idx_high = np.where(X[:, node.feature] > node.break_point)
            return [(X[idx_low], y[idx_low]), (X[idx_high], y[idx_high])]
        else:
            return [(np.array([]), np.array([]))]

    def create_node(self, X, y, name, parent):
        value = np.mean(y)
        error = np.sum((y - value) ** 2)
        mse = error / len(y)
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
        if break_points.size == 0:
            break_points = [None]
        return break_points

    def evaluate_break_point(self, Xf, y, break_point=None):
        if break_point:
            low = y[np.where(Xf <= break_point)]
            high = y[np.where(Xf > break_point)]
            error_low = np.sum((low - np.mean(low)) ** 2)
            error_high = np.sum((high - np.mean(high)) ** 2)
            error = error_low + error_high
        else:
            error = np.sum((y - np.mean(Xf)) ** 2)
        return error

    def predict(self, X):
        n_samples = X.shape[0]
        y = np.zeros(n_samples)
        for i in range(n_samples):
            y[i] = self.predict_sample(X[i])
        return y

    def predict_sample(self, sample):
        node = self.root
        while not node.is_leaf:
            node = node.split(sample)
        return node.value

    def get_feature_importance(self, X, y):
        error = np.mean((y - self.predict(X)) ** 2)
        n_features = X.shape[1]
        feature_importances = np.zeros(n_features)
        for feature in range(n_features):
            X_temp = X[:, feature]
            X_shuf = np.random.shuffle(X_temp.copy())
            X[:, feature] = X_shuf
            feature_importances[feature] = np.mean((y - self.predict(X)) ** 2) - error
        return feature_importances / np.sum(feature_importances)


class RandomForest:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X, y):
        all_idxs = np.arange(X.shape[0])
        for i in range(self.n_trees):
            tree = DecisionTree()
            sample_idxs = np.random.choice(all_idxs, X.shape[0], replace=True)
            tree.fit(X[sample_idxs], y[sample_idxs])
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_trees))
        for idx, tree in enumerate(self.trees):
            predictions[:, idx] = tree.predict(X)
        return predictions
