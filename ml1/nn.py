import numpy as np


class Layer:
    def __init__(self, input_size, output_size, bias=True):
        self.W = np.random.rand(output_size, input_size)
        self.b = np.zeros(output_size)
        self.grad = None

    def __call__(self, X):
        self.grad = X
        return np.dot(X, self.W.T) + self.b

    def update(self, loss, learning_rate):
        pass


if __name__ == "__main__":
    Xbatch = np.random.rand(10, 10)
    l = Layer(10, 4)
    pred = l(Xbatch)
    print(pred)
