import numpy as np
import tensorflow as tf
from functools import reduce, partial
from tensorflow.python.util import object_identity
from upstride.type2.tf.keras import layers as type2_layers

def log10(x):
  base = 10.
  return tf.math.log(x) / tf.math.log(base)


def calc_accuracy(y_true, y_pred):
  y_true = tf.math.argmax(tf.convert_to_tensor(y_true, tf.float32), axis=-1)
  y_pred = tf.math.argmax(tf.convert_to_tensor(y_pred, tf.float32), axis=-1)
  return tf.math.reduce_mean(tf.cast((tf.math.equal(y_true, y_pred)), dtype=tf.float32))


def count_trainable_params(model):
  """
  Count the number of trainable parameters of tf.keras model
  Args
      model: tf.keras model
  return
      Total number ot trainable parameters
  """
  weights = model.trainable_weights
  total_trainable_params = int(sum(np.prod(p.shape.as_list()) for p in object_identity.ObjectIdentitySet(weights)))
  return total_trainable_params


def _count_flops_conv2d(layer):
  if layer.data_format == "channels_first":
    input_channels = layer.input_shape[1]
    output_channels, h, w, = layer.output_shape[1:]
  elif layer.data_format == "channels_last":
    input_channels = layer.input_shape[3]
    h, w, output_channels = layer.output_shape[1:]
  w_h, w_w = layer.kernel_size

  n_mult = h * w * output_channels * input_channels * w_h * w_w
  n_add = w_h * w_w * input_channels * h * w * output_channels

  flops = n_mult + n_add

  if layer.use_bias:
    flops += output_channels * h * w

  return int(flops)

def _count_flops_conv2d_type2(layer):
  if layer.data_format == "channels_first":
    input_channels = layer.input_shape[1]
    output_channels, h, w, = layer.output_shape[1:]
  elif layer.data_format == "channels_last":
    input_channels = layer.input_shape[3]
    h, w, output_channels = layer.output_shape[1:]
  w_h, w_w = layer.kernel_size

  n_mult = h * w * output_channels * input_channels * w_h * w_w
  n_add = w_h * w_w * input_channels * h * w * output_channels

  op = 8 * (n_mult + n_add)
  input_shape = 8 * np.prod(layer.input_shape[1:])
  kernel_shape = 8 * np.prod(layer.kernel_size)
  output_shape = 12 * np.prod(layer.output_shape[1:])

  flops = op + input_shape + kernel_shape + output_shape

  if layer.use_bias:
    flops += output_channels * h * w * 4

  return int(flops)

def _count_flops_relu(layer):
  """ Dev note : current tensorflow profiler say ReLU doesn't cost anything...
  """
  # 2 operations per component : compare and assign
  return reduce(lambda x, y: x*y, layer.output_shape[1:]) * 2

# TODO refactor the below functions for any algebra
def _count_flops_relu_type2(layer):
  return _count_flops_relu(layer) * 4

def _count_flops_hard_sigmoid(layer):
  return _count_flops_relu(layer) * 2 # relu + 1 addition 1 division for each element

def _count_flops_hard_sigmoid_type2(layer):
  return _count_flops_hard_sigmoid(layer) * 4

def _count_flops_hard_swish(layer):
  return _count_flops_hard_sigmoid(layer) * 2 # hard_sigmoid + 1 multiplication

def _count_flops_hard_swish_type2(layer):
  return _count_flops_hard_swish(layer) * 4

def _count_flops_maxpool2d(layer):
  return layer.pool_size[0] * layer.pool_size[1] * reduce(lambda x, y: x*y, layer.output_shape[1:])

def _count_flops_maxpool2d_type2(layer):
  return _count_flops_maxpool2d(layer) * 4

def _count_flops_globalmaxpool2d(layer):
  return reduce(lambda x, y: x*y, layer.input_shape[1:])

def _count_flops_globalmaxpool2d_type2(layer):
  return _count_flops_globalmaxpool2d(layer) * 4

def _count_flops_globalavgpool2d(layer):
  return reduce(lambda x, y: x*y, layer.input_shape[1:])

def _count_flops_globalavgpool2d_type2(layer):
  return _count_flops_globalavgpool2d(layer) * 4

def _count_flops_add(layer):
  return reduce(lambda x, y: x*y, layer.output_shape[1:])

def _count_flops_add_type2(layer):
  return reduce(lambda x, y: x*y, layer.output_shape[1:]) * 4

def _count_flops_multiply(layer):
  return reduce(lambda x, y: x*y, layer.output_shape[1:])

def _count_flops_multiply_type2(layer):
  return reduce(lambda x, y: x*y, layer.output_shape[1:]) * 4

def _count_flops_bn(layer):
  return reduce(lambda x, y: x*y, layer.output_shape[1:]) * 2

def _count_flops_bn_type2(layer):
  return _count_flops_bn(layer) * 4

def _count_flops_dense(layer):
  n_mult = layer.input_shape[1] * layer.output_shape[1]
  n_add = layer.input_shape[1] * layer.output_shape[1]
  flops = n_mult + n_add
  if layer.use_bias:
    flops += layer.output_shape[1]
  return flops

def _count_flops_dense_type2(layer):
  input_shape = layer.input_shape[1]
  output_shape = layer.output_shape[1]
  q_op = 8 * input_shape * output_shape
  q_kernel_shape = 8 * input_shape * output_shape # for better verbosity 
  q_input = 8 * input_shape
  q_output = 12 * output_shape

  flops = q_op + q_kernel_shape + q_input + q_output
  if layer.use_bias:
    flops += layer.output_shape[1] * 4
  return flops

def _count_flops_depthwiseconv2d(layer):
  if layer.data_format == "channels_first":
    input_channels = layer.input_shape[1]
    output_channels, h, w, = layer.output_shape[1:]
  elif layer.data_format == "channels_last":
    input_channels = layer.input_shape[3]
    h, w, output_channels = layer.output_shape[1:]
  w_h, w_w = layer.kernel_size

  n_neurons_output = h * w * output_channels
  n_mult = w_h * w_w * n_neurons_output
  n_add = (w_h * w_w - 1) * n_neurons_output

  flops = n_mult + n_add

  if layer.use_bias:
    flops += n_neurons_output

  return int(flops)

def _count_flops_depthwiseconv2d_type2(layer):
  if layer.data_format == "channels_first":
    input_channels = layer.input_shape[1]
    output_channels, h, w, = layer.output_shape[1:]
  elif layer.data_format == "channels_last":
    input_channels = layer.input_shape[3]
    h, w, output_channels = layer.output_shape[1:]
  w_h, w_w = layer.kernel_size

  n_neurons_output = h * w * output_channels
  n_mult = w_h * w_w * n_neurons_output
  n_add = (w_h * w_w - 1) * n_neurons_output

  op = 8 * (n_mult + n_add)
  input_shape = 8 * np.prod(layer.input_shape[1:])
  kernel_shape = 8 * np.prod(layer.kernel_size)
  output_shape = 12 * np.prod(layer.output_shape[1:])

  flops = op + input_shape + kernel_shape + output_shape

  if layer.use_bias:
    flops += n_neurons_output * 4

  return int(flops)

def format_flops(flops):
  if flops // 10**9 > 0:
    return str(round(flops / 10.**9, 2)) + ' GFLOPs'
  elif flops // 10**6 > 0:
    return str(round(flops / 10.**6, 2)) + ' MFLOPs'
  elif flops // 10**3 > 0:
    return str(round(flops / 10.**3, 2)) + ' KFLOPs'
  else:
    return str(round(flops), 2) + ' FLOPs'

def count_flops_efficient(model):
  flops = 0

  # Not all the activations are present in keras layers. 
  # TODO add new layers to the engine for both tensorflow and upstride.
  map_activation = {
    "relu": _count_flops_relu,
    "hard_sigmoid": _count_flops_hard_sigmoid,
    "hard_swish": _count_flops_hard_swish,
  }

  map_layer_to_count_fn = {
      tf.keras.layers.Conv2D: _count_flops_conv2d,
      tf.keras.layers.ReLU: _count_flops_relu,
      tf.keras.layers.MaxPooling2D: _count_flops_maxpool2d,
      tf.keras.layers.GlobalMaxPooling2D: _count_flops_maxpool2d,
      tf.keras.layers.GlobalAveragePooling2D: _count_flops_globalavgpool2d,
      tf.keras.layers.Add: _count_flops_add,
      tf.keras.layers.Multiply: _count_flops_multiply,
      tf.keras.layers.BatchNormalization: _count_flops_bn,
      tf.keras.layers.Dense: _count_flops_dense,
      tf.keras.layers.DepthwiseConv2D: _count_flops_depthwiseconv2d
  }

  for layer in model.layers:
    if type(layer) in map_layer_to_count_fn:
      flops += map_layer_to_count_fn[type(layer)](layer)
    if type(layer) == tf.keras.layers.Activation:
      flops += map_activation[layer.activation.__name__](layer)
        
  return format_flops(flops)

def count_flops_efficient_type2(model):
  flops = 0

  # Not all the activations are present in keras layers. 
  # TODO add new layers to the engine for both tensorflow and upstride.
  map_activation = {
    "relu": _count_flops_relu_type2,
    "hard_sigmoid": _count_flops_hard_sigmoid_type2,
    "hard_swish": _count_flops_hard_swish_type2,
  }

  map_layer_to_count_fn = {
      type2_layers.Conv2D: _count_flops_conv2d_type2,
      tf.keras.layers.ReLU: _count_flops_relu_type2,
      tf.keras.layers.MaxPooling2D: _count_flops_maxpool2d_type2,
      tf.keras.layers.GlobalMaxPooling2D: _count_flops_maxpool2d_type2,
      tf.keras.layers.GlobalAveragePooling2D: _count_flops_globalavgpool2d_type2,
      tf.keras.layers.Add: _count_flops_add_type2,
      tf.keras.layers.Multiply: _count_flops_multiply_type2,
      tf.keras.layers.BatchNormalization: _count_flops_bn_type2,
      type2_layers.Dense: _count_flops_dense_type2,
      type2_layers.DepthwiseConv2D: _count_flops_depthwiseconv2d_type2
  }

  for layer in model.layers:
    if type(layer) in map_layer_to_count_fn:
      print(layer)
      flops += map_layer_to_count_fn[type(layer)](layer)
    if type(layer) == tf.keras.layers.Activation:
      flops += map_activation[layer.activation.__name__](layer)

  return format_flops(flops)

def count_flops(model):
  """
  Count the number of FLOPS of tf.keras model
  Args
      model: tf.keras model
  return
      Total number of FLOPS
  """
  session = tf.compat.v1.Session()
  graph = tf.Graph()
  with graph.as_default():
    with session.as_default():
      # Make temporary clone of our model under the graph
      temp_model = tf.keras.models.clone_model(model)
      run_meta = tf.compat.v1.RunMetadata()
      opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
      flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
  # To avoid flops accumulation for multiple run, reset the graph
  del graph
  return flops.total_float_ops


def information_density(model):
  """
  Calculate accuracy per M params introduced in this paper (https://arxiv.org/pdf/1605.07678.pdf)
  """
  def metric(y_true, y_pred):
    # Counting parameters in millions
    total_params = count_trainable_params(model) / 1.0e6
    accuracy = calc_accuracy(y_true, y_pred) * 100.0
    info_density = accuracy / total_params
    return info_density
  return metric


def net_score(model, alpha=2.0, beta=0.5, gamma=0.5):
  """
  Calculate custom evaluation metrics for energy efficient model by considering accuracy, computational cost and
  memory footprint, introduced in this paper (https://arxiv.org/pdf/1806.05512.pdf)
  Args
      model: tf keras model
      alpha: coefficient that controls the influence of accuracy
      beta:  coefficient that controls the influence of architectural complexity
      gamma: coefficient that controls the influence of computational complexity

  """
  def metric(y_true, y_pred):
    # Counting parameters in millions
    total_params = count_trainable_params(model) / 1.0e6
    # Counting MACs in Billions (assuming 1 MAC = 2 FLOPS)
    total_MACs = ((count_flops(model) / 2.0) / 1.0e9)
    accuracy = calc_accuracy(y_true, y_pred) * 100.0
    score = 20 * log10(tf.math.pow(accuracy, alpha) / (tf.math.pow(total_params, beta) * tf.math.pow(total_MACs, gamma)))
    return score
  return metric


# custom metrices  by extending tf.keras.metrics.Metric
class InformationDensity(tf.keras.metrics.Metric):
  """
  Calculate accuracy per M params introduced in this paper (https://arxiv.org/pdf/1605.07678.pdf)

  """

  def __init__(self, model, name='information_density', **kwargs):
    super(InformationDensity, self).__init__(name=name, **kwargs)
    self.model = model
    self.info_density = self.add_weight(name='info_density', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):

    info_density = information_density(self.model)(y_true, y_pred)

    self.info_density.assign_add(info_density)

  def result(self):
    return self.info_density

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.info_density.assign(0.)


class NetScore(tf.keras.metrics.Metric):
  """
      Calculate custom evaluation metrics for energy efficient model by considering accuracy, computational cost and
      memory footprint, introduced in this paper (https://arxiv.org/pdf/1806.05512.pdf)
      Args
          model: tf keras model
          alpha: coefficient that controls the influence of accuracy
          beta:  coefficient that controls the influence of architectural complexity
          gamma: coefficient that controls the influence of computational complexity

      """

  def __init__(self, model, alpha=2.0, beta=0.5, gamma=0.5, name='net_score', **kwargs):
    super(NetScore, self).__init__(name=name, **kwargs)
    self.model = model
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.net_score = self.add_weight(name='netscore', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):

    score = net_score(self.model)(y_true, y_pred)

    self.net_score.assign_add(score)

  def result(self):
    return self.net_score

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.net_score.assign(0.)
