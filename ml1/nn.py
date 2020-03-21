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

    def update(self):
        for layer in self.layers:
            layer.update()


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

    def update(self, learning_rate=0.01):
        self.weight -= learning_rate * (self.weight_grad)
        self.bias -= learning_rate * (self.bias_grad)

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


class LogSoftmax:
    def __call__(self, X):
        eX = np.exp(X)
        row_sums = np.sum(eX, axis=1).reshape(eX.shape[0], 1)
        self.softmax = eX / row_sums
        return np.log(self.softmax)

    def backward(self, loss):
        n_batches = self.softmax.shape[0]
        n_outputs = self.softmax.shape[1]
        softmax_array = -1 * np.repeat(self.softmax, n_outputs, axis=1).reshape(
            n_batches, n_outputs, -1
        )
        idxs = np.broadcast_to(np.arange(n_outputs), (n_batches, n_outputs)).reshape(
            n_batches * n_outputs
        )
        batch_idxs = np.repeat(np.arange(n_batches), n_outputs)
        softmax_array[batch_idxs, idxs, idxs] += 1
        return softmax_array


class NLLLoss:
    def __call__(self, y_pred, y_true):
        self.loss = np.zeros(y_pred.shape)
        idx = np.arange(y_pred.shape[0])
        self.loss[idx, y_true] = -1 / y_pred.shape[0]
        return np.mean(-y_pred[idx, y_true])

    def backward(self, loss=None):
        return self.loss


class MSE:
    def __call__(self, y_pred, y_true):
        self.loss = -2 * (y_true - y_pred)
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, loss=None):
        return (self.loss,)


def validate(net, X_valid, y_valid, loss_func):
    preds = net(X_valid)
    loss = loss_func(preds, y_valid)
    acc = accuracy(preds, y_valid)
    return loss, acc


def accuracy(preds, y_valid):
    idxs = np.argmax(preds, axis=1)
    accuracy = np.sum(idxs == y_valid)
    return accuracy


def step(net, X_batch, y_batch, loss_func):
    preds = net(X_batch)
    loss = loss_func(preds, y_batch)
    acc = accuracy(preds, y_batch)
    loss_grad = loss_func.backward()
    net.backward(loss_grad)
    net.update()
    return loss, acc


def fit(net, X_train, y_train, X_valid, y_valid, loss_func, batch_size=100, n_epochs=1):
    set_size = X_train.shape[0]
    idxs = np.arange(set_size)
    n_batches = set_size // batch_size
    for epoch in range(n_epochs):
        np.random.shuffle(idxs)
        loss = 0
        acc = 0
        for batch in range(n_batches):
            start = batch * batch_size
            batch_idxs = idxs[start : start + batch_size]
            X_batch = X_train[batch_idxs]
            y_batch = y_train[batch_idxs]
            b_loss, b_acc = step(net, X_batch, y_batch, loss_func)
            loss += b_loss
            acc += b_acc
        loss = loss / n_batches
        acc = acc / n_batches
        val_loss, val_acc = validate(net, X_valid, y_valid, loss_func)
        print(
            f"{epoch + 1} acc:{acc:.3f}, loss:{loss:.3f}, val_acc:{val_acc:.3f}, val_loss:{val_loss:.3f}"
        )


if __name__ == "__main__":
    ls = nn.LogSoftmax(dim=1)
    nlll = nn.NLLLoss()
    X = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], requires_grad=True)
    y = torch.tensor([0, 1], dtype=torch.int64)
    logs = ls(X)
    loss = nlll(logs, y)
    loss.backward()
    Xn = X.data.numpy()
    yn = y.data.numpy()
    my_ls = LogSoftmax()
    my_nlll = NLLLoss()
    my_logs = my_ls(Xn)
    print("LOG SOFTMAX", my_logs)
    my_loss = my_nlll(my_logs, yn)
    print("GRAD", X.grad)
    b = my_nlll.backward()
    print(b)
    c = my_ls.backward(b)
    print(c)
