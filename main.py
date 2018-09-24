import tensorflow as tf
import os
import argparse

try:
    from model import *
    from Data import dataset
except ImportError:
    from hyperencoder.model import *
    from hyperencoder.Data import dataset

# make log dir
if not os.path.exists('logs'):
    os.mkdir('logs')

# idk what this is doing
map(os.unlink, (os.path.join('logs', f) for f in os.listdir('logs')))


if __name__ == '__main__':

    # argument parser
    def parse_args():
        parser = argparse.ArgumentParser()

        # hyperparameters
        parser.add_argument('--epoch', type=int, default=25, help='Epoch to train [25]')
        parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for Adam [0.0002]')
        parser.add_argument('--beta1', type=float, default=0.5, help='Momentum term of Adam [0.5]')
        parser.add_argument('--batch_size', type=int, default=64, help='The size of the batch images [64]')

        # data input/output
        parser.add_argument('--x_height', type=int, default=227,
                            help='The height of the input image [227]')
        parser.add_argument('--x_width', type=int, default=None,
                            help='The width of the input image. If None, same value as x_height [None]')
        parser.add_argument('--y_height', type=int, default=64,
                            help='The height of the image to be generated [64]')
        parser.add_argument('--y_width', type=int, default=None,
                            help='The width of the image to be generated. If None, same value as y_height. [None]')
        parser.add_argument('--keep_grayscale', type=bool, required=False,
                            help='Flag to specify if grayscale images should be included in the training dataset')

        # directories
        parser.add_argument('--dataset_dir', type=str, required=True,
                            help='The directory where the dataset are stored')
        parser.add_argument('--dataset_name', type=str, default=None,
                            help='Name of the dataset')

        parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'checkpoint'),
                            help='Directory name to save the checkpoints [./checkpoint]')
        parser.add_argument('--sample_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'samples'),
                            help='Directory name to save the image samples [./samples]')
        parser.add_argument('--log_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'logs'),
                            help='Directory name to save the logs [./logs]')

        # train flag
        parser.add_argument('--train', type=bool, default=False, help='True for training, False for testing [False]')

        # for encoder
        # TODO: maybe the same as --checkpoint_dir
        parser.add_argument('--model_path', type=str, default=os.path.join(os.path.dirname(__file__), 'models'),
                            help='Enter the path for the model to use for testing [models]')
        parser.add_argument('--tf_record_path', type=str, default=os.path.join(os.path.dirname(__file__), 'tf_record_path'),
                            help='Enter the path for the TF Record File to use for training [./tf_record_path]')

        # define the size of the embedding
        parser.add_argument('--embed_size', dest='embed_size', default=100, type=int,
                            help='Size of the embedding layer [100]')

        args = parser.parse_args()

        # verify path
        dp = args.dataset_dir
        if not os.path.isdir(dp):
            raise ValueError('%s is not a directory' % dp)

        # adjust data
        if args.x_width is None:
            args.x_width = args.x_height
        if args.y_width is None:
            args.y_width = args.y_height
        if args.dataset_name is None:
            args.dataset_name = os.path.basename(dp.strip('\\'))

        # make dirs
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        if not os.path.exists(args.sample_dir):
            os.mkdir(args.sample_dir)
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)

        return args

    args = parse_args()
    datapath = args.dataset_dir
    batch_size = args.batch_size
    x_shape = args.x_height, args.x_width, 3
    y_shape = args.y_height, args.y_width, 3
    dsname = args.dataset_name

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:

        if args.train:
            kwargs = vars(args)
            ds = dataset(datapath, batch_size, x_shape[:2], y_shape[:2], name=dsname)

            # build model
            encoder = HyperEncoder(sess=sess,
                                   data=ds,
                                   x_shape=x_shape,
                                   y_shape=y_shape,
                                   embed_dim=args.embed_size,
                                   **kwargs)

            # train model
            encoder.train(learning_rate=kwargs['learning_rate'],
                          beta1=kwargs['beta1'],
                          epochs=kwargs['epoch'],
                          sample_dir=kwargs['sample_dir'],
                          log_dir=kwargs['log_dir'])


