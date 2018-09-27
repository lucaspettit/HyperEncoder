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

    op = config['operation']
    if op == 'train':
        # create directories if needed
        dirs = [
            config['resources']['checkpoint_dir'],
            config['resources']['sample_dir'],
            config['resources']['log_dir']
        ]

    elif op in ('freeze', 'load'):
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
        ]
    else:
        raise ValueError('Unrecognized operation "{}"'.format(op))

    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    return config


config = load_config()

op = config['operation']

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True

with tf.Session(config=run_config) as sess:
    if op == 'train':

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

    elif op == 'freeze':
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

        # build model
        encoder = HyperEncoder.build(**init_kwargs)

        # freeze model
        encoder.freeze(config['resources']['model_dir'])

    elif op == 'load':
        import cv2
        import matplotlib.pyplot as plt

        frozen_filename ='%s.pb' % config['model']['name']
        full_frozen_filename = os.path.join(config['resources']['model_dir'], frozen_filename)

        encoder = HyperEncoder.load_frozen(sess, full_frozen_filename)

        x_dim = tuple(config['model']['x_dim'][:2])
        img = cv2.imread('res/cat.jpg')
        img = cv2.resize(img, x_dim, interpolation=cv2.INTER_CUBIC)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        x_shape = tuple(config['model']['x_dim'])
        x_size = np.product(np.array(x_shape))
        padding = [np.zeros(x_size, dtype=np.uint8).reshape(x_shape) for _ in range(63)]
        input = [img] + padding

        embs = encoder.encode(input)
        print(embs[0])

        _img = encoder.decode(embs)[1]

        _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
        plt.imshow(_img)
        plt.show()



print('done!')