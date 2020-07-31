import tensorflow as tf

import tic_tac_toe.logic as game_logic

NUM_ROWS = game_logic.NUM_ROWS
NUM_COLS = game_logic.NUM_COLS
NUM_ACTIONS = game_logic.NUM_ACTIONS

HIDDEN_SIZE = 64
LEARNING_RATE = .01


class Model(object):
  def __init__(self):
    state = tf.placeholder(tf.float32, (None, 2, NUM_ROWS, NUM_COLS))
    value = tf.placeholder(tf.float32, (None,))
    action_probabilities = tf.placeholder(tf.float32, (None, NUM_ACTIONS))
    legal_actions = tf.placeholder(tf.bool, (None, NUM_ACTIONS))

    x = tf.reshape(state, (-1, 2 * NUM_ROWS * NUM_COLS))
    x = tf.layers.dense(x, HIDDEN_SIZE, activation=tf.nn.relu)
    # x = tf.layers.dense(x, HIDDEN_SIZE, activation=tf.nn.relu)
    # x = tf.layers.dense(x, HIDDEN_SIZE, activation=tf.nn.relu)

    v_predictions = tf.layers.dense(x, 1, activation=tf.nn.tanh)
    v_loss = tf.reduce_mean(tf.losses.mean_squared_error(
        labels=tf.expand_dims(value, axis=1), predictions=v_predictions))

    p_logits = tf.layers.dense(x, NUM_ACTIONS)
    p_predictions = tf.nn.softmax(p_logits)
    p_predictions *= tf.to_float(legal_actions)
    p_predictions /= tf.expand_dims(tf.reduce_sum(p_predictions, axis=1),
                                    axis=1)
    p_weights = tf.to_float(action_probabilities > 0)
    p_loss = tf.reduce_mean(tf.losses.log_loss(
        labels=action_probabilities, predictions=p_predictions,
        weights=p_weights))

    loss = v_loss + p_loss
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(loss, tf.train.get_global_step())

    summary = tf.summary.merge_all()

    self.state = state
    self.value = value
    self.action_probabilities = action_probabilities
    self.legal_actions = legal_actions
    self.loss = loss
    self.train_op = train_op
    self.v_predictions = v_predictions
    self.p_predictions = p_predictions
    self.summary = summary
