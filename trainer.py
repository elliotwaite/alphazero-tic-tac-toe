import numpy as np
import tensorflow as tf

import tic_tac_toe.logic as game_logic

NUM_ACTIONS = game_logic.NUM_ACTIONS
LOG_DIR = 'log_dir'
BATCH_SIZE = 128


class Trainer(object):
  def __init__(self, model):
    self.model = model
    self.saver = tf.train.Saver()
    self.sess = tf.Session()
    self.summary_writer = tf.summary.FileWriter(LOG_DIR, self.sess.graph)
    self.sess.run(tf.global_variables_initializer())

  @staticmethod
  def _get_batch(training_pool):
    batch = {
      'state': [],
      'value': [],
      'action_probabilities': [],
      'legal_actions': []}
    sampled_indexes = np.random.choice(
        len(training_pool), size=min(BATCH_SIZE, len(training_pool)),
        replace=False)
    for i in sampled_indexes:
      batch['state'].append(training_pool[i][0])
      batch['value'].append(training_pool[i][1])
      batch['action_probabilities'].append(training_pool[i][2])
      batch['legal_actions'].append(training_pool[i][2] > 0)
    return batch

  def train(self, training_pool):
    batch = self._get_batch(training_pool)
    feed_dict = {self.model.state: batch['state'],
                 self.model.value: batch['value'],
                 self.model.action_probabilities: batch['action_probabilities'],
                 self.model.legal_actions: batch['legal_actions']}
    _, loss = self.sess.run([self.model.train_op, self.model.loss],
                            feed_dict=feed_dict)
    print('Loss: %f\n' % loss)

  def predict(self, state, legal_actions):
    feed_dict = {self.model.state: [state],
                 self.model.legal_actions: [legal_actions]}
    value, actions_probabilities = self.sess.run(
        [self.model.v_predictions, self.model.p_predictions],
        feed_dict=feed_dict)

    return value[0], actions_probabilities[0]
