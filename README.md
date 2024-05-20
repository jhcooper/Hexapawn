# CISC 681 Programming Project 3: Hexapawn

This project is a solution to CISC 681 Programming Assignment 3. It consists of three main components: `game.py`, `hexapawn.py`, and `neuralNetwork.py`.

## Dependencies

This project requires the following Python libraries:

- numpy

You can install these dependencies using pip:

```bash
pip install numpy
```
## game.py

The `game.py` file contains the logic for the game of Hexapawn. It includes functions to generate all possible states of the game, build a policy table, and determine the result of a game given a state and an action. This file contains the answers to parts 1 and 2 of the assignment.

## hexapawn.py

The `hexapawn.py` file implements the Hexapawn class. This class trains a neural network to learn the optimal policy for the game of Hexapawn. The Hexapawn class uses the NeuralNetwork class from `neuralNetwork.py` to create a neural network with the specified parameters. It then trains this network on the initial state of the game. This file contains the answer to part 6 of the assignment.

## neuralNetwork.py

The `neuralNetwork.py` file contains the implementation of the NeuralNetwork and Layer classes, which are used to create and train a neural network for the Hexapawn game. It also contains classify and update_weights methods for the NeuralNetwork class, which are used to classify inputs and update weights using backpropagation. Additionally, it contains the implementation of the sigmoid and ReLU activation functions, as well as their derivatives. This file contains the answers to parts 3, 4, and 5 of the assignment.

## Running the Project

To run the project, execute the `hexapawn.py` file. This will train the neural network and then test it on all possible states of the game. The output will show the predicted action and the optimal action for each state.