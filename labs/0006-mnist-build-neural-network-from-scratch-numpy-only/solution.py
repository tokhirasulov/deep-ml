import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

    
    def forward(self, X):
        score1 = X @ self.W1 + self.b1
        relu = np.maximum(0, score1)
        score2 = relu @ self.W2 + self.b2
        shifted = score2 - np.max(score2, axis=1, keepdims=True)
        exponents = np.exp(shifted)
        row_sums = np.sum(exponents, axis=1, keepdims=True)
        softmax = exponents / row_sums
        
        self.score1 = score1
        self.relu = relu

        return softmax
    
    def backward(self, X, y, probs):
        total = 0.0
        for i in range(len(y)):
            correct_prob = probs[i][y[i]]
            total += -np.log(correct_prob + 1e-12)
        loss = total / len(y)

        one_hot = np.eye(self.output_size)[y] 
        dz2 = (probs - one_hot) / len(y)
        grad_W2 = self.relu.T @ dz2
        grad_b2 = np.sum(dz2, axis=0)

        dh = dz2 @ self.W2.T
        dz1 = dh * (self.score1 > 0)
        grad_W1 = X.T @ dz1
        grad_b1 = np.sum(dz1, axis=0)


        self.W1 -= self.lr * grad_W1
        self.W2 -= self.lr * grad_W2
        self.b1 -= self.lr * grad_b1
        self.b2 -= self.lr * grad_b2

        return loss
    
    def train_step(self, X, y):
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)