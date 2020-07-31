import numpy as np

NUM_ROWS = 3
NUM_COLS = 3
NUM_ACTIONS = 9


def get_start_state():
  # The board state will be 2 layers of 3x3 values. The first layer will
  # be the current player's pieces, and the second layer will be the
  # opponent's pieces. The values will be 1 where pieces are placed,
  # and 0 otherwise.
  return np.zeros([2, NUM_ROWS, NUM_COLS], dtype=np.int8)


def get_legal_actions(state):
  # Returns an ndarray of boolean values, True if the action is legal,
  # and False if it is not.
  played_pieces = state.sum(axis=0).astype(bool)
  return np.logical_not(played_pieces).flatten()


def get_next_state(state, action):
  # Create a new state, add the move for the current player, then reverse the
  # first dimension to swap which player is the current player.
  next_state = np.copy(state)
  row = action // NUM_COLS
  col = action % NUM_COLS
  next_state[0, row, col] = 1
  return next_state[::-1]


def get_game_outcome(state):
  # Check if the current player or the opponent won.
  for player in range(2):
    pieces = state[player]
    # Check for vertical, horizontal, and diagonal wins.
    if (np.any(pieces.sum(axis=0) == 3) or
        np.any(pieces.sum(axis=1) == 3) or
        pieces.trace() == 3 or
        np.flip(pieces, axis=0).trace() == 3):
      return 1 if player == 0 else -1

  # Check if the game is a draw (if 9 pieces have been played and there hasn't
  # been a winner yet).
  if state.sum() == 9:
    return 0

  # Otherwise the game has not finished.
  return None
