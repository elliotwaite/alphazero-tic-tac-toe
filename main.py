import collections

import mcts
import trainer
import tic_tac_toe.model as game_model
import tic_tac_toe.display as game_display

NUM_TRAIN_STEPS = 100000
GAMES_PER_TRAIN_STEP = 1
REPLAY_BUFFER_SIZE = 50


def main():

  # Initialize the model.
  model = game_model.Model()
  model_trainer = trainer.Trainer(model)

  # Create a training pool of self-play game data. The training pool will
  # be a deque of tuples. Each tuple will contain: (a game state, the game
  # outcome, and the action probabilities for that game state that were
  # calculated using MCTS).
  buffer = collections.deque(maxlen=REPLAY_BUFFER_SIZE)

  # Run `NUM_TRAIN_STEPS` training steps.
  for _ in range(NUM_TRAIN_STEPS):

    # Play `GAMES_PER_TRAIN_STEP` games and add the generated training data
    # to the replay_buffer.
    for _ in range(GAMES_PER_TRAIN_STEP):
      game_data = mcts.play_a_game(model_trainer)
      game_display.display_game_data(game_data)
      print(f'Value function outputs: {[v for _, v, _ in game_data]}')
      buffer.extend(game_data)

    # Train the model.
    model_trainer.train(buffer)


if __name__ == '__main__':
    main()
