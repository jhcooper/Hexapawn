"""
Contains Answers to Parts 1 and 2


game.py holds all logic for the declaration of Hexapawn, as well as the implementation of the minimax algorithm
"""


def toDee(s):
    """
    Given a state, turns it into a 2D array.

    Parameters:
        - s (list): Current state of the game
    Returns:
        - list: 2D array representation of the state
    """
    return [s[1:4], s[4:7], s[7:]]


def toState(board, player):
    """
    Given a 2D array and turn, turns it back into a state.

    Parameters:
        - board (list): 2D array representation of the state
        - player (int): Player whose turn it is to move (0 or 1)
    Returns:
        - list: Current state of the game
    """
    return [player] + board[0] + board[1] + board[2]


def toMove(state):
    """
    Given a state, returns the player whose turn it is to move.

    Parameters:
        - state (list): Current state of the game

    Returns:
        - int: Player whose turn it is to move (0 or 1)
    """
    return state[0]


def actions(state):
    """
    Given a state, returns the set of legal moves.

    Parameters:
        - state (list): Current state of the game
    Returns:
        - list: Set of legal moves
    """
    player = toMove(state)
    moves = []
    board = toDee(state)

    if player == 0:
        for i in range(3):
            for j in range(3):
                if board[i][j] == 1:
                    if i > 0 and board[i - 1][j] == 0:
                        moves.append(["advance", i, j])
                    if j > 0 and i > 0 and board[i - 1][j - 1] == -1:
                        moves.append(["captureLeft", i, j])
                    if j < 2 and i > 0 and board[i - 1][j + 1] == -1:
                        moves.append(["captureRight", i, j])
    else:
        for i in range(3):
            for j in range(3):
                if board[i][j] == -1:
                    if i < 2 and board[i + 1][j] == 0:
                        moves.append(["advance", i, j])
                    if j > 0 and i < 2 and board[i + 1][j - 1] == 1:
                        moves.append(["captureLeft", i, j])
                    if j < 2 and i < 2 and board[i + 1][j + 1] == 1:
                        moves.append(["captureRight", i, j])
    return moves


def result(state, action):
    """
    Given a state and action, returns the new state.

    Parameters:
        - state (list): Current state of the game
        - action (list): Action to take

    Returns:
        - list: New state of the game
    """
    player = toMove(state)
    board = toDee(state)
    row, col = action[1], action[2]

    if action[0] == "advance":
        new_row = row - 1 if player == 0 else row + 1
        board[new_row][col] = board[row][col]
    elif action[0] == "captureLeft":
        new_row, new_col = (row - 1, col - 1) if player == 0 else (row + 1, col - 1)
        board[new_row][new_col] = board[row][col]
    elif action[0] == "captureRight":
        new_row, new_col = (row - 1, col + 1) if player == 0 else (row + 1, col + 1)
        board[new_row][new_col] = board[row][col]

    board[row][col] = 0
    next_player = 1 if player == 0 else 0
    return toState(board, next_player)


def isTerminal(state):
    """
    Given a state, returns True if the game is over, False otherwise.

    Parameters:
        - state (list): Current state of the game

    Returns:
        - bool: True if the game is over, False otherwise
    """
    board = toDee(state)
    if 1 in board[0] or -1 in board[2]:
        return True
    return len(actions(state)) == 0


def utility(state):
    """
    Given a terminal state, returns the utility of the state.

    Parameters:
        - state (list): Current state of the game
    Returns:
        - int: Utility of the state
    """
    board = toDee(state)
    if 1 in board[0]:
        return 1
    elif -1 in board[2]:
        return -1
    return 0


def minimax(state):
    """
    Given a state, returns the best action for the current player using the minimax algorithm.

    Parameters:
        - state (list): Current state of the game
    Returns:
        - list: Best action for the current player [action, row, col]

    """
    if toMove(state) == 0:
        val, action = maxValue(state)
    else:
        val, action = minValue(state)
    return action


def minValue(state):
    """
    Given a state, returns the minimum value of the state and the best action.

    Parameters:
        - state (list): Current state of the game
    Returns:
        - tuple: (int, list) representing the minimum value and best action
    """
    if isTerminal(state):
        return utility(state), None
    v1 = float("inf")
    best_action = None
    for action in actions(state):
        v2, _ = maxValue(result(state, action))
        if v2 < v1:
            v1 = v2
            best_action = action
    return v1, best_action


def maxValue(state):
    """
    Given a state, returns the maximum value of the state and the best action.

    Parameters:
        - state (list): Current state of the game
    Returns:
        - tuple: (int, list) representing the maximum value and best action
    """
    if isTerminal(state):
        return utility(state), None
    v1 = float("-inf")
    best_action = None
    for action in actions(state):
        v2, _ = minValue(result(state, action))
        if v2 > v1:
            v1 = v2
            best_action = action
    return v1, best_action


def buildPolicyTable(state):
    """
    Given a start state, builds the [value, action] table for all possible states until the game is over.

    Parameters:
        - state (list): Current state of the game
    Returns:
        - dict: Policy table for all possible states until the game is over
    """
    policy = {}
    states = generateStates(state, [])  # Generate all possible game states
    for s in states:
        action = minimax(s)
        v1 = utility(s)
        v2 = utility(result(s, action)) if action else v1
        policy[str(s)] = (v1, v2, action)
    return policy


def generateStates(state, all_states):
    """
    Given a state, returns all possible states that can be reached from that state.

    Parameters:
        - state (list): Current state of the game
        - all_states (list): List of all possible states
    Returns:
        - list: List of all possible states that can be reached from the given state
    """
    if isTerminal(state):
        return all_states
    for action in actions(state):
        new_state = result(state, action)
        if new_state not in all_states:
            all_states.append(new_state)
            generateStates(
                new_state, all_states
            )  # Recursively generate all possible states
    return all_states
