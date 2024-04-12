from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_weights(n_inputs, n_neurons, input_ranges):
    weights = []
    for _ in range(n_neurons):
        weights.append(np.random.uniform(low=input_ranges[0], high=input_ranges[1], size=(n_inputs + 1, 1)))
    return weights

def perceptron(X, y_true, eta, epochs, n_inputs, n_neurons, input_ranges):
    m, n = X.shape
    weights = initialize_weights(n_inputs, n_neurons, input_ranges)
    n_miss_list = []

    for epoch in range(epochs):
        n_miss = 0
        for idx, x_i in enumerate(X):
            x_i = np.insert(x_i, 0, 1).reshape(-1, 1)
            layer_outputs = []
            for w in weights:
                z = np.dot(x_i.T, w)
                a = sigmoid(z)
                layer_outputs.append(a)
            y = np.mean(layer_outputs)
            if abs(y - y_true[idx]) > 0.01:
                for i, w in enumerate(weights):
                    error = y_true[idx] - layer_outputs[i]
                    w += eta * error * x_i
                n_miss += 1
        n_miss_list.append(n_miss)

    return weights, n_miss_list

def plot_decision_boundary_network(X, y_true, weights):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(X[:, 0][y_true == 0], X[:, 1][y_true == 0], "r^")
    plt.plot(X[:, 0][y_true == 1], X[:, 1][y_true == 1], "bs")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title("Perceptron Algorithm")
    
    mean_weights = np.mean(weights, axis=0)
    m = -mean_weights[1] / mean_weights[2]
    c = -mean_weights[0] / mean_weights[2]
    x1 = [min(X[:, 0]), max(X[:, 0])]
    x2 = m * np.array(x1) + c
    plt.plot(x1, x2, 'y-')
    
    plt.show()


def predict(X, weights):
    predictions = []
    for x_i in X:
        x_i = np.insert(x_i, 0, 1).reshape(-1, 1)
        layer_outputs = []
        for w in weights:
            z = np.dot(x_i.T, w)
            a = sigmoid(z)
            layer_outputs.append(a)
        y = np.mean(layer_outputs)
        predictions.append(1 if y >= 0.5 else 0)
    return np.array(predictions)


# Usage example:
X, y_true = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)

theta, miss_l = perceptron(X, y_true, 0.5, 15, n_inputs=2, n_neurons=2, input_ranges=(-4, 4))
plt.plot(miss_l)
plot_decision_boundary_network(X, y_true, theta)

# Generate new test data
X_test, y_test = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
y_pred = predict(X_test, theta)

print("Predicted labels:", y_pred)
print("True labels:", y_test)
