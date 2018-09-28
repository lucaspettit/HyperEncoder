import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.python.ops import nn_ops
import tensorflow.contrib.slim as slim
import numpy as np
import time
import cv2
import re
import os
from math import floor

try:
    from hyperencoder.Data import dataset
except ImportError as e:
    from Data import dataset


class HyperEncoder(object):
    def __init__(self, name, sess):

        self.graph = tf.get_default_graph()
        self._loaded = False
        self._name = name
        self.sess = sess

        self.data = None
        self._is_training = None
        self.batch_size = None
        self.x_shape = None
        self.y_shape = None
        self.emb_dim = None
        self.loss = None
        self.img_sum = None
        self.loss_sum = None
        self.embed_layer = None
        self.output_layer = None
        self.decode_loss = None
        self.saver = None
        self._checkpoint_counter = None
        self._input_padding = None
        self._embed_padding = None

    @classmethod
    def build(cls, sess, data_controller, checkpoint_dir, name='HyperEncoder', batch_size=64, x_shape=(227, 227, 3),
              y_shape=(128, 128, 3), embed_dim=128):

        new_class = cls(name, sess)

        new_class._checkpoint_dir = checkpoint_dir
        new_class._is_training = True
        new_class.data = data_controller
        new_class.batch_size = batch_size
        new_class.x_shape = [x for x in x_shape]
        new_class.y_shape = [y for y in y_shape]
        new_class.emb_dim = embed_dim
        new_class.loss = None
        new_class.img_sum = None
        new_class.loss_sum = None
        new_class.decode_loss = None
        new_class.embed_layer = None
        new_class.output_layer = None
        new_class.input_layer = None
        new_class.saver = None
        new_class._checkpoint_counter = 0

        new_class.build_network()

        ckpt_loaded, checkpoint_counter = new_class._load(True)

        if not ckpt_loaded:
            new_class._checkpoint_counter = checkpoint_counter
            print(' [!] Load FAILED')
        else:
            new_class._checkpoint_counter = checkpoint_counter
            new_class._loaded = True
            print(' [*] Load SUCCESS')

        new_class._loaded = True
        return new_class


    @classmethod
    def load_frozen(cls, sess, frozen_filename, name='HyperEncoder'):
        # create the new class
        new_class = cls(name, sess)
        new_class._is_training = False
        new_class.data = None
        new_class.loss = None
        new_class.img_sum = None
        new_class.loss_sum = None
        new_class.decode_loss = None
        new_class.embed_layer = None
        new_class.output_layer = None
        new_class.input_layer = None
        new_class.saver = None
        new_class._checkpoint_counter = 0

        # load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
        with tf.gfile.GFile(frozen_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # import the graph_def into the default_graph
        graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name=name)

        #for op in graph.get_operations():
        #    print(op.name)

        new_class.x = graph.get_tensor_by_name('%s/x:0' % name)
        new_class.embed_layer = graph.get_tensor_by_name('%s/encoder/embedding/Relu:0' % name)
        new_class.output_layer = graph.get_tensor_by_name('%s/decoder/y:0' % name)

        new_class.batch_size = new_class.x.shape[0]
        new_class.x_shape = new_class.x.shape[1:]
        new_class.y_shape = new_class.output_layer.shape[1:]
        new_class.emb_dim = new_class.embed_layer.shape[1]

        x_shape = np.array(new_class.x_shape)
        #x_size = np.product(x_shape)
        new_class._input_padding = [np.zeros(x_shape, dtype=np.uint8)] * new_class.batch_size
        new_class._embed_padding = np.zeros((new_class.batch_size, new_class.emb_dim))

        return new_class

    def init_decoder(self, x, scope):
        # decode variables
        initializer = tf.random_normal_initializer(stddev=0.02)
        strides = [1, 2, 2, 1]

        with tf.name_scope(scope):
            # project (4, 4, 1024)
            decode_project_w = tf.get_variable(shape=[x.get_shape()[-1], 4 * 4 * 1024], initializer=initializer,
                                               name='project_w')
            decode_project_b = tf.get_variable(shape=[4 * 4 * 1024], initializer=initializer,
                                               name='project_b')
            decode_project = tf.matmul(x, decode_project_w) + decode_project_b
            decode_project = tf.reshape(decode_project, [self.batch_size, 4, 4, 1024],
                                        name='project')
            _decode_project = tf.nn.relu(decode_project)

            # decode conv 1
            decode_conv1_w = tf.get_variable(shape=[5, 5, 512, 1024], initializer=initializer,
                                             name='conv1_w')
            decode_conv1_b = tf.get_variable(shape=[512], initializer=tf.constant_initializer(0.0),
                                             name='conv1_b')
            decode_conv1 = tf.nn.conv2d_transpose(_decode_project, decode_conv1_w,
                                                  output_shape=[self.batch_size, 8, 8, 512], strides=strides)
            decode_conv1 = tf.reshape(tf.nn.bias_add(decode_conv1, decode_conv1_b), decode_conv1.get_shape(),
                                      name='conv1')
            _decode_conv1 = tf.nn.relu(decode_conv1)

            # decode conv 2
            decode_conv2_w = tf.get_variable(shape=[5, 5, 256, 512], initializer=initializer,
                                             name='conv2_w')
            decode_conv2_b = tf.get_variable(shape=[256], initializer=tf.constant_initializer(0.0),
                                             name='conv2_b')
            decode_conv2 = tf.nn.conv2d_transpose(_decode_conv1, decode_conv2_w,
                                                  output_shape=[self.batch_size, 16, 16, 256], strides=strides)
            decode_conv2 = tf.reshape(tf.nn.bias_add(decode_conv2, decode_conv2_b), decode_conv2.get_shape(),
                                      name='conv2')
            _decode_conv2 = tf.nn.relu(decode_conv2)

            # decode conv 3
            decode_conv3_w = tf.get_variable(shape=[5, 5, 128, 256], initializer=initializer,
                                             name='conv3_w')
            decode_conv3_b = tf.get_variable(shape=[128], initializer=tf.constant_initializer(0.0),
                                             name='conv3_b')
            decode_conv3 = tf.nn.conv2d_transpose(_decode_conv2, decode_conv3_w,
                                                  output_shape=[self.batch_size, 32, 32, 128], strides=strides)
            decode_conv3 = tf.reshape(tf.nn.bias_add(decode_conv3, decode_conv3_b),
                                      decode_conv3.get_shape(), name='conv3')
            _decode_conv3 = tf.nn.relu(decode_conv3)

            # decode conv 4
            conv4_w = tf.get_variable(shape=[5, 5, 64, 128], initializer=initializer, name='conv4_w')
            conv4_b = tf.get_variable(shape=[64], initializer=tf.constant_initializer(0.0), name='conv4_b')
            conv4 = tf.nn.conv2d_transpose(_decode_conv3, conv4_w, output_shape=[self.batch_size, 64, 64, 64],
                                           strides=strides)
            conv4 = tf.reshape(tf.nn.bias_add(conv4, conv4_b), conv4.get_shape(), name='conv4')
            _conv4 = tf.nn.relu(conv4)

            # decode conv 5
            conv5_w = tf.get_variable(shape=[5, 5, 3, 64], initializer=initializer, name='conv5_w')
            conv5_b = tf.get_variable(shape=[3], initializer=tf.constant_initializer(0.0), name='conv5_b')
            conv5 = tf.nn.conv2d_transpose(_conv4, conv5_w, output_shape=[self.batch_size, 128, 128, 3],
                                           strides=strides)
            conv5 = tf.reshape(tf.nn.bias_add(conv5, conv5_b), conv5.get_shape(), name='conv5')

            output_layer = tf.nn.tanh(conv5, name='y')

        return output_layer

    def init_encoder(self, input, scope):

        with tf.name_scope(scope):
            conv1 = slim.conv2d(input, 96, [11, 11], 4, padding='VALID', scope='conv1')  # phi 1  conv1
            max1 = slim.max_pool2d(conv1, [3, 3], 2, padding='VALID', scope='max1')  # phi 1 / max1

            conv1a = slim.conv2d(max1, 256, [4, 4], 4, padding='SAME', scope='conv1a')  # phi 2 / conv1a

            conv2 = slim.conv2d(max1, 256, [5, 5], 1, scope='conv2')  # phi 3 / conv2
            max2 = slim.max_pool2d(conv2, [3, 3], 2, padding='VALID', scope='max2')  # phi 3 / max2
            conv3 = slim.conv2d(max2, 384, [3, 3], 1, scope='conv3')  # phi 3 / conv3

            conv3a = slim.conv2d(conv3, 256, [2, 2], 2, padding='SAME', scope='conv3a')  # phi 3 / conv3a

            conv4 = slim.conv2d(conv3, 384, [3, 3], 1, scope='conv4')  # phi 5 / conv4
            conv5 = slim.conv2d(conv4, 256, [3, 3], 1, scope='conv5')  # phi 5 / conv5
            pool5 = slim.max_pool2d(conv5, [3, 3], 2, padding='SAME', scope='pool5')  # phi 5 / pool5

            concat_feat = tf.concat([conv1a, conv3a, pool5], 3)
            conv_all = slim.conv2d(concat_feat, 192, [1, 1], 1, padding='VALID', scope='conv_all')  # phi 5 / oonv_all

            shape = int(np.prod(conv_all.get_shape()[1:]))
            fc_encode1 = slim.fully_connected(tf.reshape(tf.transpose(conv_all, [0, 3, 1, 2]), [-1, shape]), 3072,
                                              scope='fc_full')  # phi 5 / fc_full

            fc_encode2 = slim.fully_connected(fc_encode1, 512, scope='fc_full2')  # phi 5 / fc_full2

            fc_embed = slim.fully_connected(fc_encode2, self.emb_dim, scope='embedding')  # phi 5 / embedding

            self.graph.add_to_collection('embedding', fc_embed)

            return fc_embed

    def build_network(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):

            # with self.graph.as_default():
            # this is the true labels
            y_dim = [self.batch_size] + self.y_shape
            self.y_ = tf.placeholder(tf.float32, y_dim, name='y_')

            # input layer
            x_dim = [self.batch_size] + self.x_shape
            self.x = tf.placeholder(tf.float32, x_dim, name='x')

            self.embed_layer = self.init_encoder(self.x, scope='encoder')
            self.output_layer = self.init_decoder(self.embed_layer, 'decoder')

            self.saver = tf.train.Saver()

    def _eval(self, batch, epoch, sample_dir, sample_set):
        # sample results
        x, y = batch
        res = self.sess.run(self.output_layer, feed_dict={self.x: x})

        y_ = [np.array(r) for r in res[:2]]
        sample = np.vstack(y[:2])
        sample_ = np.vstack(y_)
        sample = np.hstack((sample, sample_))

        sample_model_dir = os.path.join(sample_dir, self.name)
        if not os.path.isdir(sample_model_dir):
            os.makedirs(sample_model_dir)

        sample_filename = os.path.join(sample_model_dir, '%d_%s_sample.jpg' % (epoch, sample_set))
        cv2.imwrite(sample_filename, sample)

        print('sample saved at %s' % sample_filename)

    def train(self, learning_rate, beta1, epochs, sample_dir, log_dir):

        if not self._is_training:
            raise AssertionError('Model is not initialized for training')

        print(' [*] Training [%s]' % self.data.name)

        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y_, self.output_layer, scope='mse_loss'))
        train_step = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.loss)

        test_eval = self.data.next_batch(training=False)
        training_eval = self.data.next_batch(training=True)

        sample_dir = os.path.join(sample_dir, self.name)
        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)

        # log writers & summaries
        with tf.name_scope('summaries'):
            img_y = self.data.denormalize_image(self.output_layer[:3])
            img_y = tf.reverse(img_y, axis=[-1])

            img_tensor = tf.concat(img_y, 1)
            img_sum = tf.summary.image('Decoded Image', img_tensor)

            loss_sum = tf.summary.scalar('Loss Summary', self.loss)

            emb_mean = tf.reduce_mean(self.embed_layer)
            emb_std = tf.sqrt(tf.reduce_mean(tf.square(self.embed_layer - emb_mean)))
            embed_hist_sum = tf.summary.histogram('Embed3 Layer Histogram', self.embed_layer)

        # make writer path
        print(' [*] Initializing writers')
        train_writer_path = os.path.join(log_dir, self.name, 'train')
        test_writer_path = os.path.join(log_dir, self.name, 'test')
        for path in (train_writer_path, test_writer_path):
            if not os.path.isdir(path):
                os.makedirs(path)

        # make writer
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(train_writer_path, self.sess.graph)
        test_writer = tf.summary.FileWriter(test_writer_path)
        test_sum_counter = 0
        train_sum_counter = 0
        train_step_count = 0

        self.sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):

            # train on batches
            for batch_count in range(self.data.num_batches):

                if train_step_count % 100 == 0:
                    batch_xs, batch_ys = self.data.next_batch(training=False)
                    output, summary_str = self.sess.run([self.output_layer, merged],
                                                        feed_dict={self.x: batch_xs, self.y_: batch_ys})

                    test_writer.add_summary(summary_str, test_sum_counter)

                    # grab the first image in batch and convert it back to [0, 255]
                    y = [self.data.denormalize_image(img) for img in output[:3]]
                    y_ = [self.data.denormalize_image(img) for img in batch_ys[:3]]

                    sample_image = np.hstack((np.vstack(y), np.vstack(y_)))

                    sample_filename = os.path.join(sample_dir, '%d_%d_%d_sample.jpg' % (
                    self._checkpoint_counter, epoch, test_sum_counter))
                    cv2.imwrite(sample_filename, sample_image)
                    test_sum_counter += 1

                # get next batch
                batch_xs, batch_ys = self.data.next_batch(training=True)

                # run train step
                start_t = time.time()
                _, summary_str = self.sess.run([train_step, merged],
                                               feed_dict={self.x: batch_xs, self.y_: batch_ys})
                end_t = time.time()

                # record summaries & display info
                train_writer.add_summary(summary_str, train_sum_counter)

                err = self.loss.eval({self.x: batch_xs, self.y_: batch_ys})

                print('Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f'
                      % (epoch, batch_count, self.data.num_batches, end_t - start_t, err))

                train_step_count += 1
                train_sum_counter += 1
                self._checkpoint_counter += 1

                if np.mod(self._checkpoint_counter, 500) == 2:
                    self.save(self._checkpoint_counter)

    def encode(self, images):
        """ Handler to Encode an image into a vector
        :param images: list of numpy matrixes (images) - image dim must be equivalent to input size
        :return: numpy matrix (embeddings)
        """

        i = 0
        N = len(images)
        embs = None

        while True:
            end = min(N, i + self.batch_size)
            batch = images[i: end]

            size = end - i
            if size < self.batch_size:
                batch += self._input_padding[:self.batch_size - size]

            if embs is None:
                embs = self.sess.run(self.embed_layer, feed_dict={self.x: batch})
            else:
                _embs = self.sess.run(self.embed_layer, feed_dict={self.x: batch})
                embs = np.vstack((embs, _embs))

            i += self.batch_size

            if i >= N - 1:
                break

        return embs

    def decode(self, embeddings):
        """ Handler to Decode a vector into an image
        :param x: list of numpy matrixes (vectors) - n = number of vectors, m = embedding size
        :return: numpy matrix or list of numpy matrixes (images)
        """
        def denormalize(img):
            _img = img + 1.0
            _img = _img * (255.0 / 2.0)
            return _img.astype(np.uint8)

        i = 0
        N = len(embeddings)
        imgs = []
        while True:
            end = min(N, i + self.batch_size)
            batch = embeddings[i: end]

            size = end - i
            if size < self.batch_size:
                batch += self._embed_padding[: self.batch_size - size]

            _imgs = self.sess.run(self.output_layer, feed_dict={self.embed_layer: batch})
            imgs += [denormalize(_imgs[i]) for i in range(size)]

            i += self.batch_size
            if i >= N - 1:
                break

        return imgs

    @property
    def name(self):
        return self._name

    def save(self, step):
        model_name = 'HyperEncoder.model'
        checkpoint_model_dir = os.path.join(self._checkpoint_dir, self.name)
        if not os.path.exists(checkpoint_model_dir):
            os.makedirs(checkpoint_model_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_model_dir, model_name),
                        global_step=step)

    def _load(self, clear_devices=True):
        if not self._loaded:
            print(' [*] reading checkpoints...')
            checkpoint_model_dir = os.path.join(self._checkpoint_dir, self.name)

            ckpt = tf.train.get_checkpoint_state(checkpoint_model_dir)

            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_model_dir, ckpt_name))
                counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
                print(" [*] Success to read {}".format(ckpt_name))
                return True, counter
            else:
                print(" [*] Failed to find a checkpoint")
                return False, 0

    def freeze(self, freeze_dir):

        # for node in [m.values() for m in self.sess.graph.get_operations()]:
        #    print(node)

        output_node_names = ['x', 'encoder/embedding/Relu', 'decoder/y']

        self._load(self._checkpoint_dir)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess,
            self.graph.as_graph_def(),
            output_node_names
        )

        if not os.path.isdir(freeze_dir):
            os.makedirs(freeze_dir)
        with tf.gfile.GFile(os.path.join(freeze_dir, '%s.pb' % self.name), 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('%d ops in the final graph.' % len(output_graph_def.node))

        return output_graph_def
