import numpy as np

import tic_tac_toe.logic as game_logic

NUM_ROWS = game_logic.NUM_ROWS
NUM_COLS = game_logic.NUM_COLS
NUM_ACTIONS = game_logic.NUM_ACTIONS
VERTICAL_BAR = u'\u2502'
HORIZONTAL_BAR = (u'\u2500\u2500\u2500\u253c'
                  u'\u2500\u2500\u2500\u253c'
                  u'\u2500\u2500\u2500')
BOARD_LINES = [VERTICAL_BAR, VERTICAL_BAR,
               '\n' + HORIZONTAL_BAR + '\n',
               VERTICAL_BAR, VERTICAL_BAR,
               '\n' + HORIZONTAL_BAR + '\n',
               VERTICAL_BAR, VERTICAL_BAR, '\n']
# RED = '\033[31m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
# BRIGHT_RED = '\033[91m'
BRIGHT_YELLOW = '\033[93m'
BRIGHT_BLUE = '\033[94m'
BRIGHT_MAGENTA = '\033[95m'
END_COLOR = '\033[0m'
X_SYMBOL = BRIGHT_YELLOW + u' \u2715 ' + END_COLOR
O_SYMBOL = u' \u25ef '


def build_board(pieces):
  # Interleave the list of pieces (3 char length strings) with the board
  # lines strings.
  interleaved = [val for pair in zip(pieces, BOARD_LINES) for val in pair]
  return ''.join(interleaved)


def row_of_boards(boards):
  # Combine multiple boards horizontally into a single row.
  combined_boards = ''
  for lines in zip(*[board.split('\n') for board in boards]):
    combined_boards += '    '.join(lines) + '\n'
  return combined_boards


def probability_str(probability):
  if probability >= .75:
    color = BRIGHT_MAGENTA
  elif probability >= .5:
    color = MAGENTA
  elif probability >= .25:
    color = BRIGHT_BLUE
  else:
    color = BLUE
  if probability == 1:
    num_str = '1.0'
  else:
    num_str = (str(probability) + '0')[1:4]
  return color + num_str + END_COLOR


def probabilities_board(probabilities):
  # Create a board of probability values.
  probabilities = [probability_str(p) if p else '   ' for p in probabilities]

  return build_board(probabilities)


def state_board(state, current_player_is_xs=True):
  # Create a board from the state.

  # Convert the state to a single board of values, 1 for the current player's
  # pieces, 2 for the opponent's pieces, and 0 for empty.
  pieces = (state * np.array([[[1]], [[2]]])).sum(axis=0).flatten()

  # Convert the pieces to display symbols.
  p_1 = X_SYMBOL if current_player_is_xs else O_SYMBOL
  p_2 = O_SYMBOL if current_player_is_xs else X_SYMBOL
  pieces = [p_1 if p == 1 else p_2 if p == 2 else '   ' for p in pieces]

  return build_board(pieces)


def probabilities_and_state_board(probabilities, state, invert=False):
  # Build both boards then merge them so that they are displayed next to
  # each other.
  boards = [probabilities_board(probabilities), state_board(state, invert)]
  return row_of_boards(boards)


def display_game_data(training_data):
  # Build boards for list of training data, usually for a single game. This
  # will return two rows of boards, the top row being the probabilities,
  # and the second row being the game states.
  state_list, _, probabilities_list = zip(*training_data)
  probabilities_boards = [probabilities_board(probabilities)
                          for probabilities in probabilities_list]
  state_boards = [state_board(state, current_player_is_xs=i % 2 == 0)
                  for i, state in enumerate(state_list)]
  return row_of_boards(probabilities_boards) + row_of_boards(state_boards)
