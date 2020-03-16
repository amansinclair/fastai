import numpy as np
import torch.nn as nn
import torch


class Layer:
    def __init__(self, input_size, output_size):
        self.weight = np.random.rand(output_size, input_size)
        self.bias = np.zeros(output_size)
        self.inputs = None
        self.batch_size = None
        self.weight_grad = np.zeros((output_size, input_size))
        self.bias_grad = np.zeros(output_size)

    def __call__(self, X):
        self.inputs = X.copy()
        self.batch_size = X.shape[0]
        return np.dot(X, self.weight.T) + self.bias

    def backward(self, loss):
        "loss.shape - > batch_size x output_size"
        self.bias_grad = np.mean(loss, axis=0)
        # for i in range(self.batch_size):
        # self.weight_grad += loss[i] * self.inputs
        # self.weight_grad = self.weight_grad / self.batch_size

    def update(self, learning_rate):
        self.W += learning_rate * (self.weight_grad)
        self.b += learning_rate * (self.bias_grad)


class ReLU:
    def __call__(self, X):
        self.input = X.copy()
        X[X < 0] = 0
        return X

    def backward(self, loss):
        self.input[self.input > 0] = 1
        self.input[self.input < 0] = 0
        return self.input * loss


class MSE:
    def __call__(self, y_true, y_pred):
        self.loss = -2 * (y_true - y_pred)
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, loss=None):
        return self.loss


if __name__ == "__main__":
    net = nn.Sequential(nn.Linear(3, 1))
    net.zero_grad()
    my_net = Layer(3, 1)
    my_net.weight = net[0].weight.data.numpy()
    my_net.bias = net[0].bias.data.numpy()
    X = np.array([[0, 1, 2], [0, 1, 2]], dtype="float32")
    y = np.array([[0.1], [1.2]], dtype="float32")
    Xt = torch.Tensor(X)
    yt = torch.Tensor(y)
    crit = nn.MSELoss()
    my_crit = MSE()
    o = net(Xt)
    my_o = my_net(X)
    print(o, my_o)
    loss = crit(yt, o)
    my_loss = my_crit(y, my_o)
    print(loss, my_loss)
    loss.backward()
    my_grad = my_crit.backward()
    print(my_grad)
    my_net.backward(my_grad)
    print(net[0].weight.grad, net[0].bias.grad)
    print(my_net.bias_grad)

