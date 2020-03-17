import numpy as np
import torch.nn as nn
import torch


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, index):
        return self.layers[index]

    def backward(self, dloss):
        for layer in reversed(self.layers):
            dloss = layer.backward(dloss)


class Linear:
    def __init__(self, input_size, output_size):
        self.weight, self.bias = self.init_weights(input_size, output_size)
        self.bias = np.zeros(output_size)
        self.inputs = None
        self.weight_grad = np.zeros((output_size, input_size))
        self.bias_grad = np.zeros(output_size)

    def init_weights(self, input_size, output_size):
        std = 1.0 / (output_size ** 0.5)
        weight = np.random.uniform(-std, std, input_size * output_size).reshape(
            output_size, input_size
        )
        bias = np.random.uniform(-std, std, output_size)
        return weight, bias

    def __call__(self, X):
        self.inputs = X.copy()
        return np.dot(X, self.weight.T) + self.bias

    def backward(self, loss):
        """loss.shape -> batch_size x output_size"""
        self.bias_grad = np.mean(loss, axis=0)
        self.weight_grad = np.dot(loss.T, self.inputs) / loss.shape[0]
        loss_return = np.dot(loss, self.weight)
        return loss_return

    def update(self, learning_rate=0.00001):
        self.weight += learning_rate * (self.weight_grad)
        self.bias += learning_rate * (self.bias_grad)

    def get_weights(self, layer):
        self.weight = layer.weight.data.numpy()
        self.bias = layer.bias.data.numpy()


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
    net = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
    net.zero_grad()
    l1 = Linear(3, 2)
    l1.get_weights(net[0])
    l2 = Linear(2, 1)
    l2.get_weights(net[2])
    X = np.array([[0, 1, 2], [0, 1, 2]], dtype="float32")
    y = np.array([[0.1], [1.2]], dtype="float32")
    Xt = torch.Tensor(X)
    yt = torch.Tensor(y)
    crit = nn.MSELoss()
    my_crit = MSE()
    rl = ReLU()
    my_net = Sequential(l1, rl, l2)
    o = net(Xt)
    my_o = my_net(X)
    print(o, my_o)
    loss = crit(yt, o)
    my_loss = my_crit(y, my_o)
    print("LOSSES", loss, my_loss)
    loss.backward()
    my_grad = my_crit.backward()
    my_net.backward(my_grad)
    for i in [0, 2]:
        print("torch ", net[i].weight.grad, net[i].bias.grad)
        print("my ", my_net.layers[i].weight_grad, my_net.layers[i].bias_grad)

