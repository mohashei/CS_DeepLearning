# Copyright 2018 Mohammed Azeem Sheikh All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compressed sensing Inception model v1
"""
import argparse
import logging

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

import util
from util import override_if_not_in_args

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils


IMAGE_H = 256
IMAGE_W = 256
IMAGE_SIZE = IMAGE_H * IMAGE_W
SPARSE_SIZE = int(np.floor( 0.5 * IMAGE_H * IMAGE_W ))
REDUCED_H = 16 
REDUCED_W = 16
L1_SIZE = REDUCED_H * REDUCED_W 
OMAP = [4, 4] 
IMAP = [1, OMAP[0] * 4]
REDUCE1X1 = [2, 2]

class GraphMod():
  TRAIN = 1
  EVALUATE = 2
  PREDICT = 3

def build_signature(inputs, outputs):
  """Build the signature.

  Not using predic_signature_def in saved_model because it is replacing the
  tensor name, b/35900497.

  Args:
    inputs: a dictionary of tensor name to tensor
    outputs: a dictionary of tensor name to tensor
  Returns:
    The signature, a SignatureDef proto.
  """
  signature_inputs = {key: saved_model_utils.build_tensor_info(tensor)
                      for key, tensor in inputs.items()}
  signature_outputs = {key: saved_model_utils.build_tensor_info(tensor)
                       for key, tensor in outputs.items()}

  signature_def = signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.PREDICT_METHOD_NAME)

  return signature_def


def create_model():
  """Factory method that creates model to be used by generic task.py."""
  parser = argparse.ArgumentParser()
  #l1_size, imap, omap, reduce1x1) during preprocessing.
  parser.add_argument('--dropout', type=float, default=0.5)
  args, task_args = parser.parse_known_args()
  override_if_not_in_args('--max_steps', '6000', task_args)
  override_if_not_in_args('--batch_size', '128', task_args)
  override_if_not_in_args('--eval_set_size', '128', task_args)
  override_if_not_in_args('--eval_interval_secs', '2', task_args)
  override_if_not_in_args('--log_interval_secs', '2', task_args)
  override_if_not_in_args('--min_train_eval_rate', '2', task_args)
  return Model(args.dropout), task_args

class GraphReferences(object):
  """Holder of base tensors used for training model using common task."""

  def __init__(self):
    self.examples = None
    self.train = None
    self.global_step = None
    self.metric_updates = []
    self.metric_values = []
    self.keys = None
    self.predictions = []
    self.inputs = None

class Model(object):
  """TensorFlow model for the flowers problem."""

  def __init__(self, dropout):
    self.dropout = dropout

  def decode(self, kspace, batch_size, xspace = None):
      k = tf.cast(tf.squeeze(tf.decode_raw(kspace, tf.float64)),
                tf.float32)
      k.set_shape((None, 2 * SPARSE_SIZE))
      x = []
      if xspace is not None:
        x = tf.squeeze(tf.decode_raw(xspace, tf.float32))
        x.set_shape((None, IMAGE_SIZE))
      
      return k, x

  def build_final_layer(self,
                        embeddings,
                        hidden_layer_size=IMAGE_SIZE):
    """Adds a fully-connected layer for training with no activation function.

    Args:
      embeddings: The embedding (bottleneck) tensor.
      all_labels_count: The number of all labels including the default label.
      hidden_layer_size: The size of the hidden_layer. Roughtly, 1/4 of the
                         bottleneck tensor size.
      dropout_keep_prob: the percentage of activation values that are retained.
    Returns:
      logits: The logits tensor. This is basically the output.
    """
    with tf.name_scope('output_layer'):
      embeddings = tf.reshape(embeddings, [-1,L1_SIZE*OMAP[1]*4])
      #No activation function
      tensors = layers.fully_connected(
        embeddings, IMAGE_SIZE, activation_fn=None)
    return tensors

  #creates weights -- from MNIST tutorial
  def createWeight(self, size, Name):
    return tf.Variable(tf.truncated_normal(size, stddev=0.1),
                          name=Name)

  #creates biases -- from MNIST tutorial
  def createBias(self, size, Name):
    return tf.Variable(tf.constant(0.01,shape=size),
                          name=Name)
  #conv -- from MNIST tutorial 
  def conv2d_s1(self, x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

  #maxpool -- from MNIST tutorial
  def max_pool_3x3_s1(self, x):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],
                          strides=[1,1,1,1],padding='SAME') 
  
  def build_inception_layer(self, x, imap, omap, reduce1x1):
    """Builds an inception graph and add the necessary input & output tensors.

    Returns:
      embeddings: The embeddings tensor.
    """
    with tf.name_scope('inception_layer'):
      W_conv1_1x1_1 = self.createWeight([1,1,imap,omap],'W_conv1_1x1_1')
      b_conv1_1x1_1 = self.createBias([omap],'b_conv1_1x1_1') 
    
      W_conv1_1x1_2 = self.createWeight([1,1,imap,reduce1x1],'W_conv1_1x1_2')
      b_conv1_1x1_2 = self.createBias([reduce1x1],'b_conv1_1x1_2')
    
      W_conv1_1x1_3 = self.createWeight([1,1,imap,reduce1x1],'W_conv1_1x1_3')
      b_conv1_1x1_3 = self.createBias([reduce1x1],'b_conv1_1x1_3')

      W_conv1_3x3 = self.createWeight([3,3,reduce1x1,omap],'W_conv1_3x3')
      b_conv1_3x3 = self.createBias([omap],'b_conv1_3x3')

      W_conv1_5x5 = self.createWeight([5,5,reduce1x1,omap],'W_conv1_5x5')
      b_conv1_5x5 = self.createBias([omap],'b_conv1_5x5')
    
      W_conv1_1x1_4= self.createWeight([1,1,imap,omap],'W_conv1_1x1_4')
      b_conv1_1x1_4= self.createBias([omap],'b_conv1_1x1_4')

      conv1_1x1_1 = self.conv2d_s1(x,W_conv1_1x1_1)+b_conv1_1x1_1
      conv1_1x1_2 = tf.nn.relu(self.conv2d_s1(x,W_conv1_1x1_2)+b_conv1_1x1_2)
      conv1_1x1_3 = tf.nn.relu(self.conv2d_s1(x,W_conv1_1x1_3)+b_conv1_1x1_3)
      conv1_3x3 = self.conv2d_s1(conv1_1x1_2,W_conv1_3x3)+b_conv1_3x3
      conv1_5x5 = self.conv2d_s1(conv1_1x1_3,W_conv1_5x5)+b_conv1_5x5
      maxpool1 = self.max_pool_3x3_s1(x)
      conv1_1x1_4 = self.conv2d_s1(maxpool1,W_conv1_1x1_4)+b_conv1_1x1_4
 
    return tf.nn.relu(tf.concat([conv1_1x1_1,conv1_3x3,conv1_5x5,conv1_1x1_4], 3))

  def build_input_layer(self, inputs, onum, dropout_keep_prob = None):
    with tf.name_scope('input_layer'): 
      hidden = layers.fully_connected(inputs, onum)
      # We need a dropout when the size of the dataset is rather small.
      if dropout_keep_prob:
        hidden = tf.nn.dropout(hidden, dropout_keep_prob)
      hidden = tf.reshape(hidden, [-1, REDUCED_W, REDUCED_H, 1]) 
    return hidden

  def build_graph(self, data_paths, batch_size, graph_mod, l1_size, imap, omap, reduce1x1, dropout_prob=None):
    tensors = GraphReferences()
    is_training = graph_mod == GraphMod.TRAIN
    if data_paths:
      tensors.keys, tensors.examples = util.read_examples(
          data_paths,
          batch_size,
          shuffle=is_training,
          num_epochs=None if is_training else 2)
    else:
      tensors.examples = tf.placeholder(tf.string, name='input', shape=(None,))
    
    if graph_mod != GraphMod.PREDICT:
      with tf.name_scope('data'):
        feature_map = {
            'height': 
                tf.FixedLenFeature(shape=1, dtype=tf.int64),
            'kspace':
                tf.FixedLenFeature(shape=1, dtype=tf.string),
            'width':    
                tf.FixedLenFeature(shape=1, dtype=tf.int64),
            'xspace':
                tf.FixedLenFeature(shape=1, dtype=tf.string)
        }
        parsed = tf.parse_example(tensors.examples, features=feature_map)
        inputs, outputs = self.decode(parsed['kspace'], batch_size, xspace = parsed['xspace'])
    
    if graph_mod == GraphMod.PREDICT:
      inputs = tf.placeholder(tf.float32, shape=(IMAGE_SIZE))
      tensors.inputs = inputs 
    
    with tf.name_scope('network'):
      layer1 = self.build_input_layer(inputs, l1_size, dropout_keep_prob = dropout_prob)
      inception1 = self.build_inception_layer(layer1, imap[0], omap[0], reduce1x1[0])
      inception2 = self.build_inception_layer(inception1, imap[1], omap[1], reduce1x1[1])
      final = self.build_final_layer(inception2)  
    
    if graph_mod == GraphMod.PREDICT:
      tensors.predictions = [final]  
      return tensors

    with tf.name_scope('evaluate'):
      loss_value = loss(final, outputs)    

    if is_training:
      tensors.train, tensors.global_step = training(loss_value)
    else:
      tensors.global_step = tf.Variable(0, name='global_step', trainable=False)

    loss_updates, loss_op = util.loss(loss_value)
    #accuracy_updates, accuracy_op = util.accuracy(final, outputs)

    if not is_training:
     # tf.summary.scalar('accuracy', accuracy_op)
      tf.summary.scalar('loss', loss_op)

    tensors.metric_updates = loss_updates# + accuracy_updates
    tensors.metric_values = [loss_op]#, accuracy_op]
    return tensors

  def build_train_graph(self, data_paths, batch_size, l1_size=L1_SIZE, imap=IMAP, 
                        omap=OMAP, reduce1x1=REDUCE1X1):
    return self.build_graph(data_paths, batch_size, GraphMod.TRAIN,  
                            l1_size, imap, omap, reduce1x1, dropout_prob = self.dropout)

  def build_eval_graph(self, data_paths, batch_size, l1_size=L1_SIZE, imap=IMAP, 
                       omap=OMAP, reduce1x1=REDUCE1X1):
    return self.build_graph(data_paths, batch_size, GraphMod.EVALUATE, 
                            l1_size, imap, omap, reduce1x1)
  
  def restore_from_checkpoint(self, session, trained_checkpoint_file):
    """To restore model variables from the checkpoint file.

       The graph is assumed to consist of an inception model and other
       layers including a softmax and a fully connected layer.
    Args:
      session: The session to be used for restoring from checkpoint.
      trained_checkpoint_file: path to the trained checkpoint for the other
                               layers.
    """
    # Restore the rest of the variables from the trained checkpoint.
    trained_vars = tf.contrib.slim.get_variables_to_restore()
    trained_saver = tf.train.Saver(trained_vars)
    trained_saver.restore(session, trained_checkpoint_file)

  def build_prediction_graph(self):
    """Builds prediction graph and registers appropriate endpoints."""

    tensors = self.build_graph(None, 1, GraphMod.PREDICT, L1_SIZE, IMAP, OMAP, REDUCE1X1)
    return tensors.inputs, tensors.predictions[0]

  def export(self, last_checkpoint, output_dir):
    """Builds a prediction graph and xports the model.

    Args:
      last_checkpoint: Path to the latest checkpoint file from training.
      output_dir: Path to the folder to be used to output the model.
    """
    logging.info('Exporting prediction graph to %s', output_dir)
    with tf.Session(graph=tf.Graph()) as sess:
      # Build and save prediction meta graph and trained variable values.
      inputs, outputs = self.build_prediction_graph()
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      self.restore_from_checkpoint(sess, last_checkpoint)
      builder = saved_model_builder.SavedModelBuilder(output_dir)
      builder.add_meta_graph_and_variables(
          sess,
          tags=[tag_constants.SERVING],
          signature_def_map={
            "model": tf.saved_model.signature_def_utils.predict_signature_def(
            inputs= {"input": inputs},
            outputs= {"output": outputs})
          }
      )
      builder.save()

  def format_metric_values(self, metric_values):
    """Formats metric values - used for logging purpose."""

    # Early in training, metric_values may actually be None.
    loss_str = 'N/A'
    #accuracy_str = 'N/A'
    try:
      loss_str = '%.3f' % metric_values[0]
     # accuracy_str = '%.3f' % metric_values[1]
    except (TypeError, IndexError):
      pass

    return '%s' % (loss_str)#, accuracy_str)

def loss(final, real_img, weights=None, reg = 0):
  if weights is None:
    return tf.nn.l2_loss(final-real_img)
  else:
    return tf.nn.l2_loss(final-real_img) + reg * tf.nn.l2_loss(weights)

def training(loss_op):
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(epsilon=0.001)
    train_op = optimizer.minimize(loss_op, global_step)
    return train_op, global_step
