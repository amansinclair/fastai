import numpy as np


class Layer:
    def __init__(self, input_size, output_size):
        self.W = np.random.rand(output_size, input_size)
        self.b = np.zeros(output_size)
        self.grad = None

    def __call__(self, X):
        self.input = X.copy()
        return np.dot(X, self.W.T) + self.b

    def update(self, loss, learning_rate):
        """ loss: batch_size x output_size"""
        layer_loss = loss * self.W
        self.W += learning_rate * (self.input * loss)
        self.b += learning_rate * (loss)
        return layer_loss


class ReLU:
    def __call__(self, X):
        self.input = X.copy()
        X[X < 0] = 0
        return X

    def update(self, loss):
        self.input[self.input > 0] = 1
        self.input[self.input < 0] = 0
        return self.input * loss


class MSE:
    def __call__(self, y_true, y_pred):
        self.loss = -2 * (y_true - y_pred)
        return np.mean((y_true - y_pred) ** 2)

    def update(self):
        return self.loss


if __name__ == "__main__":
    Xbatch = np.random.rand(10, 10)
    l = Layer(10, 4)
    pred = l(Xbatch)
    print(pred)
