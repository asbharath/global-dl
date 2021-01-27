"""
This file contains everything to talk to Upstride API

Exceptionally, this file doesn't have unittest, as the functions here depends on the Alchemy platform
Maybe in the future we could add integration tests for testing the communication

"""
import requests
import tensorflow as tf
from . import metrics

SERVER_URL = 'https://alchemyback.upstride.io'

arguments = [
    [str, 'user', '', 'Username to connect to UpStride platform. If not provided then the code will not try to connect'],
    [str, 'password', '', 'Password to connect to UpStride platform'],
    [str, 'id', '', 'id of the training, provided by the server at the first post request'],
    [str, 'jwt', '', 'javascript web token for auth']
]


def _get_header(jwt):
  """ return the javascript header with jwt token for api call
  """
  return {'Authorization': "Bearer " + jwt}


def send_metric_callbacks(config):
  """create and return a keras callback to send data to the server

  Args:
      config (Dict): parsed arguments from argument_parser
  """
  def send_metric(epoch, logs):
    """
    Args:
        logs: dictionary containing  loss, accuracy, top_k_categorical_accuracy, val...

    """
    if not config['user']:
      return
    # prepare message
    json_content = []
    for key in logs:
      json_content.append({'value': float(logs[key]), 'name': key})
    # send it with jwt or basic auth security
    if config['jwt'] != '':
      headers = _get_header(config['jwt'])
      requests.put(SERVER_URL + f'/training/metric/{config["id"]}', headers=headers, json={'epoch': epoch, 'values': json_content})
    elif config['password'] and config['user']:
      auth = (config['user'], config['password'])
      requests.put(SERVER_URL + f'/training/metric/{config["id"]}', auth=auth, json={'epoch': epoch, 'values': json_content})
  return tf.keras.callbacks.LambdaCallback(on_epoch_end=send_metric)


def send_final_checkpoint():
  # TODO not yet implemented
  pass


def send_exported_model(config, archive_path):
  if config['password'] and config['user']:
    auth = (config['user'], config['password'])
    with open(archive_path, 'rb') as f:
      files = {'file_to_upload': f}
      r = requests.post(f'{SERVER_URL}/training/upload_saved_model/{config["id"]}', auth=auth, files=files)


def start_training(config):
  """ send experiment data to the server
  """
  if config['user'] and not config['id']:
    message = {'args': {}}
    for key in config:
      if key not in ['server', 'description', 'title']:
        # we don't want to set these parameters in the training parameters
        message['args'][key] = config[key]
    message['description'] = config['description']
    message['title'] = config['title']
    message['git_tag'] = '' # TODO send git tag to the server
    try:
      r = requests.post(SERVER_URL + '/training', json=message, auth=(config['user'], config['password']), timeout=10)
    except Exception as e:
      print("can't send result to the server, training continue")
    if r.status_code == 401:
      raise Exception("Provided authorization is not valid")
    config['id'] = r.json()
  return config


def send_model_info(model, config):
  """compute the number of variable and number of flops and send them to the server

  Args:
      config (Dict): argument to connect to the server
  """
  if config['password'] and config['user']:
    n_params = metrics.count_trainable_params(model)
    n_flops = metrics.count_flops_efficient(model)
    print("number of parameters:", n_params)
    print("number of flops:", n_flops)
    message = {
        'n_flops': n_flops,
        'n_params': n_params
    }
    r = requests.put(SERVER_URL + f'/training/model_info/{config["id"]}', json=message, auth=(config['user'], config['password']))
