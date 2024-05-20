"""
Contains answer to Part 6

hexapawn.py holds the implementation of the Hexepawn class, which trains a neural network to learn the optimal policy for the game.
"""

import random
from neuralNetwork import NeuralNetwork
from game import generateStates, buildPolicyTable, result


class Hexepawn:
    """
    Hexepawn class that trains a neural network to learn the optimal policy for the game.

    Attributes:
        - N (int): Number of hidden layers in the neural network
        - M (int): Number of neurons in each hidden layer
        - activation (str): Activation function for the neural network (relu or sigmoid)
        - lr (float): Learning rate for the neural network
        - nn (NeuralNetwork): Neural network object
    """

    def __init__(self, N, M, activation="sigmoid", lr=0.01):
        self.N = N
        self.M = M
        self.activation = activation
        self.lr = lr
        self.nn = NeuralNetwork(
            num_inputs=10, N=N, M=M, num_outs=9, activation=activation
        )

    def train(self, state, epochs=200):
        """
        Trains the neural network to predict the optimal move for any state in the game.

        Parameters:
            - state (list): Initial state of the game
            - epochs (int): Number of training epochs (default=200)
        """
        inputs = generateStates(state, [])
        policy = buildPolicyTable(state)
        expected = {}
        # Generate the optimal move for each possible state
        for i, s in enumerate(inputs):
            action = policy[str(s)][2]
            if action:
                expected[str(s)] = result(s, action)[1:]
            else:
                expected[str(s)] = None
        # Training set tuple
        training_data = [
            (inputs[i], expected[str(inputs[i])]) for i in range(len(inputs))
        ]

        for epoch in range(epochs):
            # Shuffle training data each epoch
            random.shuffle(training_data)

            for input_data, expected_output in training_data:
                if expected_output is not None:
                    inputs = input_data
                    outputs = self.nn.classify(inputs)
                    self.nn.update_weights(expected_output)

            print(f"Epoch {epoch + 1}/{epochs} completed")


if __name__ == "__main__":
    hexepawn = Hexepawn(N=2, M=15, activation="sigmoid")
    state = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]
    hexepawn.train(state)
    print("Training complete")
    # Test the neural network
    states = generateStates(state, [])
    policy = buildPolicyTable(state)
    right = 0
    for state in states:
        if policy[str(state)][2]:
            action = hexepawn.nn.classify(state)
            optimal = result(state, policy[str(state)][2])[1:]
            print("====================================")
            print(f"State: {state}")
            print(f"Predicted Action: {action}")

            print("Optimal Action: ", result(state, policy[str(state)][2]))
            print("====================================")
            if all(action == optimal):
                right += 1
    print(str(right) + " correct predictions out of " + str(len(states)))

"""
PART 6 EXPLANATION:

Q: Describe the architecture that you find learns to play the best game. For how many states does
it confidently suggest at least one optimal move? For how many does it confidently suggest a bad
(or illegal) move?

A: 
The architecture that I find learns to play the best game consists of a neural network with 2 hidden layers, each containing 15 neurons. 
The activation function used is the sigmoid function, and the learning rate is set to 0.01. The neural network is trained for 200 epochs 
on the initial state of the game. When asked to predict the optimal move for each possible state, the neural network is not able to confidently
suggest any optimal moves.



"""
