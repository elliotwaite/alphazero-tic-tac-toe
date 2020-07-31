import math

import numpy as np

import tic_tac_toe.logic as game_logic

NUM_SIMULATIONS_PER_MOVE = 1000
SIMULATION_EXPLORATION_LEVEL = 1


class Node(object):
  def __init__(self, state, action, nn_action_probability):
    self.state = state
    self.action = action
    self.nn_action_probability = nn_action_probability
    self.visit_count = 0
    self.total_action_value = 0
    self.average_action_value = 0
    self.children = None

  def expand_node_and_get_predicted_value(self, model):
    # Expand this node and return the predicted value of the current state.
    legal_actions = game_logic.get_legal_actions(self.state)
    value, nn_action_probabilities = model.predict(self.state, legal_actions)
    self.children = []
    for action, (action_is_legal, nn_action_probability) in enumerate(
        zip(legal_actions, nn_action_probabilities)):
      if action_is_legal:
        next_state = game_logic.get_next_state(self.state, action)
        child = Node(next_state, action, nn_action_probability)
        self.children.append(child)
    return value

  def get_child_node_from_simulation_policy(self):
    # Return the child node that the simulation should explore.
    sum_of_visit_counts = sum([c.visit_count for c in self.children])
    sqrt_of_sum_of_visit_counts = math.sqrt(sum_of_visit_counts)
    best_node = None
    max_val = -float('inf')
    for c in self.children:
      # A variant of the PUCT algorithm.
      val = c.average_action_value + (
          SIMULATION_EXPLORATION_LEVEL * c.nn_action_probability *
          sqrt_of_sum_of_visit_counts / (c.visit_count + 1))
      if val > max_val:
        max_val = val
        best_node = c
    return best_node

  def get_sampled_child_node_from_move_policy(self):
    # Sample a child node from the probability distribution that is generated
    # by dividing each child node's visit count by the sum of all the child
    # node's visit counts.
    sum_of_visit_counts = sum([c.visit_count for c in self.children])
    probabilities = [c.visit_count / sum_of_visit_counts for c in self.children]
    sampled_index = np.random.choice(len(self.children), p=probabilities)
    return self.children[sampled_index]

  def get_best_child_node_from_move_policy(self):
    # Return the child node with the highest visit count.
    best_node = None
    max_visit_count = -float('inf')
    for c in self.children:
      if c.visit_count > max_visit_count:
        max_visit_count = c.visit_count
        best_node = c
    return best_node

  def get_training_data(self):
    # Return a tuple of (state, action probabilities) for this node's state.
    # The action probabilities will be the visit count for that action
    # divided by the sum of the visit counts for all actions. Illegal actions
    # will default to having an action probability of zero.
    action_probabilities = np.zeros(game_logic.NUM_ACTIONS)
    sum_of_visit_counts = sum([c.visit_count for c in self.children])
    for c in self.children:
      action_probabilities[c.action] = c.visit_count / sum_of_visit_counts
    return self.state, action_probabilities


class Game(object):
  def __init__(self, model):
    self.model = model
    start_state = game_logic.get_start_state()
    self.root = Node(start_state, action=0, nn_action_probability=0)

    # We expand the root node to initialize the graph, but we can ignore the
    # predicted value for the root state since it won't be back propagated
    # anywhere.
    print(self.root.expand_node_and_get_predicted_value(self.model))
    print([c.nn_action_probability for c in self.root.children])

    self.training_data = []

  def is_game_over(self):
    # Check if the root node is in a terminal state. This will be True if the
    # game outcome is not None.
    game_outcome = game_logic.get_game_outcome(self.root.state)
    return game_outcome is not None

  def run_simulation(self):
    # Start at the root node and traverse down the tree, choosing the best
    # child node for simulation exploration, until we reach a leaf node.
    # Add each node we visit, including the leaf node, to `visited_nodes`.
    visited_nodes = []
    cur_node = self.root
    while cur_node.children:
      cur_node = cur_node.get_child_node_from_simulation_policy()
      visited_nodes.append(cur_node)

    # If the leaf node is a terminal state, set the leaf value to the game
    # outcome.
    game_outcome = game_logic.get_game_outcome(cur_node.state)
    if game_outcome is not None:
      leaf_value = game_outcome

    # Otherwise expand the node and set the leaf value to the predicted value.
    else:
      leaf_value = cur_node.expand_node_and_get_predicted_value(self.model)

    # Back propogate the leaf value through the visited nodes. Walk through
    # the visited nodes in reverse order, flipping the sign of the value at
    # each node so that the value added will be relative to the current player
    # of the parent node. The value should be relative to the current player
    # of the parent node because a node's action value represents the value
    # of the parent node choosing this node as its action. So the leaf
    # node's total action value will be increased by the negative of the game
    # outcome value at that leaf node because the game outcome value is
    # calculated with respect to the current player at the leaf instead of
    # the current player at its parent node.
    for node in visited_nodes[::-1]:
      leaf_value = -leaf_value
      node.visit_count += 1
      node.total_action_value += leaf_value
      node.average_action_value = node.total_action_value / node.visit_count

  def make_a_move(self):
    # Collect the training data from the current root node's state,
    # then sample a child node from the move policy and make it the new root
    # node.
    self.training_data.append(self.root.get_training_data())
    self.root = self.root.get_sampled_child_node_from_move_policy()

  def get_training_data(self):
    # Add the game outcome value to the training data then return the final
    # list. The last item in the training data is for the state just before
    # the terminal state. So it's game outcome value will be the negative of
    # the terminal state's game outcome value (since the value should be
    # relative to current player for that state). And the value will
    # continue to be flipped for each previous state, hence why the training
    # data is processed in reverse.
    game_outcome = game_logic.get_game_outcome(self.root.state)
    final_training_data = []
    for state, action_probabilities in self.training_data[::-1]:
      game_outcome = -game_outcome

      # Only add the training data where there are multiple legal moves to
      # choose from. If there is only one legal move available, only one
      # action probability will be greater than 0.
      if (action_probabilities > 0).sum() > 1:
        final_training_data.append((state, game_outcome, action_probabilities))

    return final_training_data[::-1]


def play_a_game(model):
  # Play a game, then return the training data collected from that game and the
  # final state of the game.
  game = Game(model)
  while not game.is_game_over():
    for _ in range(NUM_SIMULATIONS_PER_MOVE):
      game.run_simulation()
    game.make_a_move()

  return game.get_training_data()
