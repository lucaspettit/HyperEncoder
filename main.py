import tensorflow as tf
import os
import sys
import json

try:
    from model import *
    from Data import dataset
except ImportError:
    from hyperencoder.model import *
    from hyperencoder.Data import dataset


# make log dir
# if not os.path.exists('logs'):
#    os.mkdir('logs')

# idk what this is doing
# map(os.unlink, (os.path.join('logs', f) for f in os.listdir('logs')))

def load_config():
    # check for correct number of arguments
    if len(sys.argv) != 2:
        raise ValueError('Missing config path argument')

    # check that path is valid
    if not os.path.isfile(sys.argv[1]):
        raise ValueError('%s is not a file' % sys.argv[1])

    with open(sys.argv[1]) as f:
        config = json.load(f)

    # create default paths
    if config['resources']['model_dir'] is None:
        config['resources']['model_dir'] = os.path.join(os.path.dirname(__file__), 'models')
    if config['resources']['tf_record_dir'] is None:
        config['resources']['tf_record_dir'] = os.path.join(os.path.dirname(__file__), 'tf_record')

    # create directories if needed
    dirs = [
        config['resources']['model_dir'],
        config['resources']['tf_record_dir'],
        config['resources']['checkpoint_dir'],
        config['resources']['sample_dir'],
        config['resources']['log_dir']
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    return config

config = load_config()

datapath = config['resources']['dataset_dir']
dsname = config['data']['dataset_name']
train = config['training']['train']
keep_grayscale = config['data']['keep_grayscale_training']

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
with tf.Session(config=run_config) as sess:
    if train:

        # pack arguments for creating dataset
        data_kwargs = {
            'input_dir': config['resources']['dataset_dir'],
            'batch_size': config['training']['batch_size'],
            'x_shape': config['model']['x_dim'],
            'y_shape': config['model']['y_dim'],
            'split': config['data']['split'],
            'name': config['data']['dataset_name'],
            'keep_grayscale': config['data']['keep_grayscale_training'],
            'shuffle': config['data']['shuffle']
        }

        # pack the arguments for creating the HyperEncoder
        init_kwargs = {
            'sess': sess,
            'data_controller': None,
            'name': config['model']['name'],
            'batch_size': config['training']['batch_size'],
            'x_shape': config['model']['x_dim'],
            'y_shape': config['model']['y_dim'],
            'embed_dim': config['model']['embed_dim'],
            'checkpoint_dir': config['resources']['checkpoint_dir']
        }

        # pack arguments for training HyperEncoder
        train_kwargs = {
            'learning_rate': config['training']['learning_rate'],
            'beta1': config['training']['beta1'],
            'epochs': config['training']['epochs'],
            'sample_dir': config['resources']['sample_dir'],
            'log_dir': config['resources']['log_dir']
        }

        # build data adapter
        da = dataset(**data_kwargs)
        init_kwargs['data_controller'] = da

        # build model
        encoder = HyperEncoder.build(**init_kwargs)

        # train model
        encoder.train(**train_kwargs)

    else:
        # pack arguments for creating dataset
        data_kwargs = {
            'input_dir': config['resources']['dataset_dir'],
            'batch_size': config['training']['batch_size'],
            'x_shape': config['model']['x_dim'],
            'y_shape': config['model']['y_dim'],
            'split': config['data']['split'],
            'name': config['data']['dataset_name'],
            'keep_grayscale': config['data']['keep_grayscale_training'],
            'shuffle': config['data']['shuffle']
        }

        # pack the arguments for creating the HyperEncoder
        init_kwargs = {
            'sess': sess,
            'data_controller': None,
            'name': config['model']['name'],
            'batch_size': config['training']['batch_size'],
            'x_shape': config['model']['x_dim'],
            'y_shape': config['model']['y_dim'],
            'embed_dim': config['model']['embed_dim'],
            'checkpoint_dir': config['resources']['checkpoint_dir']
        }

        # build data adapter
        #da = dataset(**data_kwargs)
        #init_kwargs['data'] = da

        # build model
        encoder = HyperEncoder.build(**init_kwargs)

        # freeze model
        encoder.freeze(config['resources']['model_dir'])

print('done!')