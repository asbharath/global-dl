import unittest
import numpy as np
import tensorflow as tf
from training.metrics import count_flops,  count_trainable_params, InformationDensity, NetScore, count_flops_efficient


class TestMetrics(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model = tf.keras.applications.resnet.ResNet50(include_top=True, weights=None,
                                                      input_tensor=tf.keras.Input(shape=(224, 224, 3)))
    cls.y_true = [0, 1, 2, 3]
    cls.y_pred = [3, 1, 2, 3]

    cls.total_params = count_trainable_params(cls.model) / 1.0e6
    cls.total_macs = ((count_flops(cls.model) / 2.0) / 1.0e9)

  def test_information_density(self):
    acc = np.mean(np.array(self.y_true) == np.array(self.y_pred)) * 100.0

    true_info_density = acc / self.total_params

    calculated_info_density = InformationDensity(self.model)(tf.one_hot(self.y_true, 4), tf.one_hot(self.y_pred, 4))

    self.assertAlmostEqual(true_info_density, calculated_info_density.numpy(), places=3)

  def test_net_score(self):
    alpha = 2.0
    beta = 0.5
    gamma = 0.5
    acc = np.mean(np.array(self.y_true) == np.array(self.y_pred)) * 100.0

    true_net_score = 20 * np.log10(np.power(acc, alpha) / (np.power(self.total_params, beta) * np.power(self.total_macs, gamma)))

    calculated_net_score = NetScore(self.model)(tf.one_hot(self.y_true, 4), tf.one_hot(self.y_pred, 4))

    print(calculated_net_score)

    self.assertAlmostEqual(true_net_score, calculated_net_score.numpy(), places=3)


class TestCountFlops(unittest.TestCase):
  def test_conv2d_no_bias(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.Conv2D(64, (3, 3), use_bias=False)(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model)
    self.assertTrue((f-ef)/f < 1e-4)

  def test_conv2d_bias(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.Conv2D(64, (3, 3))(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model)
    self.assertTrue((f-ef)/f < 1e-4)

  def test_conv2d_strides(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2))(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model)
    self.assertTrue((f-ef)/f < 1e-4)

  def test_conv2d_padding_same(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model)
    self.assertTrue((f-ef)/f < 1e-4)
    print(f)
    print(ef)
    print((f-ef)/f)

  def test_conv2d_relu(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model)
    self.assertTrue((f-ef)/f < 1e-4)

  def test_relu(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.ReLU()(i)
    model = tf.keras.Model(i, x)
    # f = count_flops(model) # return 0...
    ef = count_flops_efficient(model)
    self.assertEqual(ef, 301056)

  def test_max_pooling(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model)
    self.assertEqual(f, ef)

  def test_dense_no_biases(self):
    i = tf.keras.layers.Input((1000), batch_size=1)
    x = tf.keras.layers.Dense(100, use_bias=False)(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model)
    # f = 400001 ??? I have no explanation for this
    # ef = 200000
    self.assertEqual(ef, 200000)

  def test_dense_biases(self):
    i = tf.keras.layers.Input((1000), batch_size=1)
    x = tf.keras.layers.Dense(100)(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model)
    self.assertEqual(ef, 200100)

  def test_depthwiseconv2d(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.DepthwiseConv2D(64, (3, 3), use_bias=False)(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model)
    self.assertTrue((f-ef)/f < 1e-3)

  def test_depthwiseconv2d_bias(self):
    i = tf.keras.layers.Input((224, 224, 3), batch_size=1)
    x = tf.keras.layers.DepthwiseConv2D(64, (3, 3), use_bias=True)(i)
    model = tf.keras.Model(i, x)
    f = count_flops(model)
    ef = count_flops_efficient(model)
    self.assertTrue((f-ef)/f < 1e-3)
