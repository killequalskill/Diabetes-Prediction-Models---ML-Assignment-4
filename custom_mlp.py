import numpy as np
class CustomMLP_4_8_1:
    def __init__(self, input_size=4, h=8, lr=0.01, epochs=3000):
        self.lr = lr
        self.epochs = epochs

        self.W1 = np.random.randn(input_size, h) / np.sqrt(input_size)
        self.b1 = np.zeros((1, h))

        self.W2 = np.random.randn(h, 1) / np.sqrt(h)
        self.b2 = np.zeros((1, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def fit(self, X, y):
        y = np.array(y).reshape(-1, 1)

        prior = y.mean()
        self.b2 = np.array([[np.log(prior / (1 - prior))]])

        for _ in range(self.epochs):
            # forward
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.relu(z1)

            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.sigmoid(z2)

            # backward
            dz2 = a2 - y
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            dz1 = np.dot(dz2, self.W2.T) * self.relu_deriv(z1)
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # update
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        return (a2 > 0.5).astype(int).flatten()