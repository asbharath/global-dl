import shutil
import tempfile
import unittest
import tensorflow as tf
import numpy as np
from training.export import export


def get_model():
  inputs = tf.keras.layers.Input(shape=(224, 224, 3))
  x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
  x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
  x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
  x = tf.keras.layers.Activation("softmax", dtype=tf.float32)(x)
  model = tf.keras.Model(inputs, x)
  return model


class TestExport(unittest.TestCase):
  def test_export(self):
    export_path = tempfile.mkdtemp()
    model = get_model()

    args = {
        'export': {
            'strategy_cloud': '',
            'tensorrt': {
                'export': False
            }
        }
    }

    export(model, export_path, args)

    image = np.ones((1, 224, 224, 3), dtype=np.float32)/127.5 - 1.

    # Then import to check if it works
    # tf.keras.models.load_model also works as working with keras, but useless because we don't need keras data
    model2 = tf.saved_model.load(export_path)

    output1 = model(image)
    output2 = model2(image)
    self.assertTrue(np.allclose(output1.numpy(), output2.numpy()))

    # model3 = tf.keras.models.load_model(export_path)
    # model3.summary()
    # for i in range(7):
    #     print(model.get_layer(index=i))

    shutil.rmtree(export_path)
