"""
Contains answers to Parts 3, 4, and 5

neuralNetwork.py holds the implementation of the NeuralNetwork and Layer classes, which are used to create and train a neural network for the Hexepawn game.

Also contains classify and update_weights methods for the NeuralNetwork class, which are used to classify inputs and update weights using backpropagation.

Also contains the implementation of the sigmoid and ReLU activation functions, as well as their derivatives.
"""

import numpy as np


class Layer:
    """
    Layer of a neural network. Contains weights, biases, and outputs for each neuron.

    Attributes:
        - N (int): Number of neurons in the layer
        - weights (np.ndarray): Weights for each neuron in the layer
        - biases (np.ndarray): Biases for each neuron in the layer
        - outputs (np.ndarray): Outputs for each neuron in the layer
    """

    def __init__(self, M, num_inputs):
        self.M = M
        self.weights = (
            2 * np.random.random((num_inputs, M)) - 1
        )  # Random weights between -1 and 1
        self.biases = 2 * np.random.random(M) - 1  # Random biases between -1 and 1
        self.outputs = np.zeros(M)


class NeuralNetwork:
    """
    Neural network class that can classify inputs and update weights using backpropagation.

    Attributes:
        - layers (list): List of Layer objects in the neural network
        - lr (float): Learning rate for the neural network
        - activation (function): Activation function for the neural network
    """

    def __init__(self, num_inputs, N, M, num_outs, activation, lr=0.1):
        """
        Initialize a neural network with N hidden layers, each with M neurons.
        """
        self.layers = []
        self.lr = lr
        if activation == "sigmoid":
            self.activation = self.sigmoid
            self.activation_prime = self.sigmoidPrime
        elif activation == "relu":
            self.activation = self.relu
            self.activation_prime = self.reluPrime

        # Input layer
        self.layers.append(Layer(M=num_inputs, num_inputs=0))

        # Hidden layers
        prev_neurons = num_inputs
        for _ in range(N):
            self.layers.append(Layer(M=M, num_inputs=prev_neurons))
            prev_neurons = M

        # Output layer
        self.layers.append(Layer(M=num_outs, num_inputs=prev_neurons))

    def classify(self, inputs):
        """
        Forward pass through the neural network to classify inputs.

        Parameters:
            - inputs (list): Input values to classify
        Returns:
            - np.ndarray: Output values of the neural network
        """
        self.layers[0].outputs = np.array(inputs)

        for i in range(1, len(self.layers)):
            prev = self.layers[i - 1].outputs
            layer = self.layers[i]
            dot = np.dot(prev, layer.weights) + layer.biases
            layer.outputs = self.activation(dot)

        return self.layers[-1].outputs

    def update_weights(self, expected):
        """
        Update the weights of the neural network using backpropagation.

        Parameters:
            - expected (np.ndarray): Expected output values for the neural network
        """
        out_layer = self.layers[-1]
        out_layer_outs = out_layer.outputs
        out_deltas = (out_layer_outs - expected) * self.activation_prime(out_layer_outs)

        prev = self.layers[-2].outputs
        out_layer.weights -= self.lr * np.outer(prev, out_deltas)
        out_layer.biases -= self.lr * out_deltas

        next_deltas = out_deltas
        for i in range(
            len(self.layers) - 2, 0, -1
        ):  # Iterate backwards through the layers
            layer = self.layers[i]
            next = self.layers[i + 1]
            prev = self.layers[i - 1].outputs
            deltas = np.dot(next.weights, next_deltas) * self.activation_prime(
                layer.outputs
            )

            layer.weights -= self.lr * np.outer(prev, deltas)
            layer.biases -= self.lr * deltas

            next_deltas = deltas

    @staticmethod
    def sigmoid(x):  # sigmoid activation function
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoidPrime(x):  # derivative of sigmoid
        return x * (1 - x)

    @staticmethod
    def relu(x):  # ReLU activation function
        return np.maximum(0, x)

    @staticmethod
    def reluPrime(x):  # derivative of ReLU
        """ """
        return np.where(x > 0, 1, 0)
