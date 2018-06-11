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

try:
    from hyperencoder.Data import dataset
except ImportError as e:
    from Data import dataset


class HyperEncoder(object):
    def __init__(self, sess, data, batch_size=64, train=False, x_shape=(227, 227, 3), y_shape=(128, 128, 3), embed_dim=128,
                 checkpoint_dir=None, **kwargs):

        self.data = data
        self.sess = sess
        self._is_training = train
        self._checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else os.path.join(os.path.dirname(__file__), 'checkpoint')

        self.batch_size = batch_size
        self.x_shape = [dim for dim in x_shape]
        self.y_shape = [dim for dim in y_shape]

        self.emb_dim = embed_dim

        self.loss = None
        self.img_sum = None
        self.loss_sum = None

        self.embed_layer = None
        self.output_layer = None
        self.decode_loss = None

        # i think this is the true labels
        self.y_ = tf.placeholder(tf.float32, [self.batch_size] + self.y_shape, name='y_')

        # input layer
        self.x = tf.placeholder(tf.float32, [self.batch_size] + self.x_shape, name='input_images')

        self.build_network()

        self.saver = tf.train.Saver()

    def init_decoder(self, x, scope):
        # decode variables
        initializer = tf.random_normal_initializer(stddev=0.02)
        strides = [1, 2, 2, 1]

        with tf.name_scope(scope):
            # project (4, 4, 1024)
            decode_project_w = tf.get_variable(shape=[x.get_shape()[-1], 4 * 4 * 1024], initializer=initializer,
                                               name='decode_project_w')
            decode_project_b = tf.get_variable(shape=[4 * 4 * 1024], initializer=initializer,
                                               name='decode_project_b')
            decode_project = tf.matmul(x, decode_project_w) + decode_project_b
            decode_project = tf.reshape(decode_project, [self.batch_size, 4, 4, 1024],
                                        name='decode_project')
            _decode_project = tf.nn.relu(decode_project)

            # decode conv 1
            decode_conv1_w = tf.get_variable(shape=[5, 5, 512, 1024], initializer=initializer,
                                             name='decode_conv1_w')
            decode_conv1_b = tf.get_variable(shape=[512], initializer=tf.constant_initializer(0.0),
                                             name='decode_conv1_b')
            decode_conv1 = tf.nn.conv2d_transpose(_decode_project, decode_conv1_w,
                                                  output_shape=[self.batch_size, 8, 8, 512], strides=strides)
            decode_conv1 = tf.reshape(tf.nn.bias_add(decode_conv1, decode_conv1_b), decode_conv1.get_shape(),
                                      name='decode_conv1')
            _decode_conv1 = tf.nn.relu(decode_conv1)

            # decode conv 2
            decode_conv2_w = tf.get_variable(shape=[5, 5, 256, 512], initializer=initializer,
                                             name='decode_conv2_w')
            decode_conv2_b = tf.get_variable(shape=[256], initializer=tf.constant_initializer(0.0),
                                             name='decode_conv2_b')
            decode_conv2 = tf.nn.conv2d_transpose(_decode_conv1, decode_conv2_w,
                                                  output_shape=[self.batch_size, 16, 16, 256], strides=strides)
            decode_conv2 = tf.reshape(tf.nn.bias_add(decode_conv2, decode_conv2_b), decode_conv2.get_shape(),
                                      name='decode_conv2')
            _decode_conv2 = tf.nn.relu(decode_conv2)

            # decode conv 3
            decode_conv3_w = tf.get_variable(shape=[5, 5, 128, 256], initializer=initializer,
                                             name='decode_conv3_w')
            decode_conv3_b = tf.get_variable(shape=[128], initializer=tf.constant_initializer(0.0),
                                             name='decode_conv3_b')
            decode_conv3 = tf.nn.conv2d_transpose(_decode_conv2, decode_conv3_w,
                                                  output_shape=[self.batch_size, 32, 32, 128], strides=strides)
            decode_conv3 = tf.reshape(tf.nn.bias_add(decode_conv3, decode_conv3_b),
                                      decode_conv3.get_shape(), name='decode_conv3')
            _decode_conv3 = tf.nn.relu(decode_conv3)

            # decode conv 4
            decode_conv4_w = tf.get_variable(shape=[5, 5, 3, 128], initializer=initializer,
                                             name='decode_conv4_w')
            decode_conv4_b = tf.get_variable(shape=[3], initializer=tf.constant_initializer(0.0),
                                             name='decode_conv4_b')
            decode_conv4 = tf.nn.conv2d_transpose(_decode_conv3, decode_conv4_w,
                                                  output_shape=[self.batch_size, 64, 64, 3], strides=strides)
            decode_conv4 = tf.reshape(tf.nn.bias_add(decode_conv4, decode_conv4_b), decode_conv4.get_shape(),
                                      name='decode_conv4')

            output_layer = tf.nn.tanh(decode_conv4)

        return [decode_project,
                decode_conv1,
                decode_conv2,
                decode_conv3,
                decode_conv4,
                output_layer]

    def init_encoder(self, input, scope='encoder'):

        with tf.name_scope(scope):
            conv1 = slim.conv2d(input, 96, [11, 11], 4, padding='VALID', scope='phi1/conv1')
            max1 = slim.max_pool2d(conv1, [3, 3], 2, padding='VALID', scope='phi1/max1')

            conv1a = slim.conv2d(max1, 256, [4, 4], 4, padding='VALID', scope='phi2/conv1a')

            conv2 = slim.conv2d(max1, 256, [5, 5], 1, scope='phi3/conv2')
            max2 = slim.max_pool2d(conv2, [3, 3], 2, padding='VALID', scope='phi3/max2')
            conv3 = slim.conv2d(max2, 384, [3, 3], 1, scope='phi3/conv3')

            conv3a = slim.conv2d(conv3, 256, [2, 2], 2, padding='VALID', scope='phi3/conv3a')

            conv4 = slim.conv2d(conv3, 384, [3, 3], 1, scope='phi6/conv4')
            conv5 = slim.conv2d(conv4, 256, [3, 3], 1, scope='phi6/conv5')
            pool5 = slim.max_pool2d(conv5, [3, 3], 2, padding='VALID', scope='phi6/pool5')

            concat_feat = tf.concat([conv1a, conv3a, pool5], 3)
            conv_all = slim.conv2d(concat_feat, 192, [1, 1], 1, padding='VALID', scope='phi6/conv_all')

            shape = int(np.prod(conv_all.get_shape()[1:]))
            fc_encode1 = slim.fully_connected(tf.reshape(tf.transpose(conv_all, [0, 3, 1, 2]), [-1, shape]), 3072,
                                              scope='phi6/fc_full1')

            fc_encode2 = slim.fully_connected(fc_encode1, 512, scope='phi6/fc_full2')
            fc_embed = slim.fully_connected(fc_encode2, self.emb_dim, scope='phi6/layer')

            return conv3a, conv1a, fc_embed

    def build_network(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):

            self.conv1a, self.conv3a, self.embed_layer = self.init_encoder(self.x, scope='encoder')
            self.output_layer = self.init_decoder(self.embed_layer, 'decoder3')[-1]

    def _eval(self, batch, epoch, sample_dir, sample_set):
        # sample results
        x, y = batch
        res = self.sess.run(self.output_layer, feed_dict={self.x: x})

        y_ = [np.array(r) for r in res[:2]]
        sample = np.vstack(y[:2])
        sample_ = np.vstack(y_)
        sample = np.hstack((sample, sample_))

        sample_model_dir = os.path.join(sample_dir, self.model_dir)
        if not os.path.isdir(sample_model_dir):
            os.makedirs(sample_model_dir)

        sample_filename = os.path.join(sample_model_dir, '%d_%s_sample.jpg' % (epoch, sample_set))
        cv2.imwrite(sample_filename, sample)

        print('sample saved at %s' % sample_filename)

    def train(self, learning_rate, beta1, epochs,
              sample_dir, log_dir):

        print(' [*] Training [%s]' % self.data.name)

        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y_, self.output_layer, scope='mse_loss'))
        train_step = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.loss)

        test_eval = self.data.next_batch(training=False)
        training_eval = self.data.next_batch(training=True)

        sample_dir = os.path.join(sample_dir, self.model_dir)
        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)

        # log writers & summaries
        with tf.name_scope('summaries'):
            conv1a_img = tf.reshape(self.conv1a[0], [96, 96, 1])
            conv3a_img = tf.reshape(self.conv3a[0], [96, 96, 1])
            img_tensor = tf.concat(self.data.denormalize_image(self.output_layer[:3]), 1)
            low_img_sum = tf.summary.image('conv1a Image', [conv1a_img])
            mid_img_sum = tf.summary.image('conv3a Image', [conv3a_img])
            img_sum = tf.summary.image('Image3 Summary', img_tensor)
            img_loss_sum = tf.summary.scalar('Image3 MSE Loss', self.loss)
            img_hist_sum = tf.summary.histogram('Image3 Histogram', self.output_layer)

            emb_mean = tf.reduce_mean(self.embed_layer)
            emb_std = tf.sqrt(tf.reduce_mean(tf.square(self.embed_layer - emb_mean)))
            emb_mean_sum = tf.summary.scalar('Embed3 Mean', emb_mean)
            emb_std_sum = tf.summary.scalar('Embed3 Std', emb_std)
            emb_max_sum = tf.summary.scalar('Embed3 Max', tf.reduce_max(self.embed_layer))
            emb_min_sum = tf.summary.scalar('Embed3 Min', tf.reduce_min(self.embed_layer))
            embed_hist_sum = tf.summary.histogram('Embed3 Layer Histogram', self.embed_layer)

        # make writer path
        print(' [*] Initializing writers')
        train_writer_path = os.path.join(log_dir, self.model_dir, 'train')
        test_writer_path = os.path.join(log_dir, self.model_dir, 'test')
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

        ckpt_loaded, checkpoint_counter = self.load(self._checkpoint_dir)
        if not ckpt_loaded:
            print(" [!] Load failed...")
            checkpoint_counter = 0
        else:
            print(' [*] Load SUCCESS')

        for epoch in range(epochs):

            # train on batches
            for batch_count in range(self.data.num_batches):

                if train_step_count % 100 == 0:
                    batch_xs, batch_ys = self.data.next_batch(training=False)
                    output, summary_str = self.sess.run([self.output_layer, merged], feed_dict={self.x: batch_xs, self.y_: batch_ys})

                    test_writer.add_summary(summary_str, test_sum_counter)

                    # grab the first image in batch and convert it back to [0, 255]
                    y = [self.data.denormalize_image(img) for img in output[:3]]
                    y_ = [self.data.denormalize_image(img) for img in batch_ys[:3]]

                    sample_image = np.hstack((np.vstack(y), np.vstack(y_)))
                    
                    sample_filename = os.path.join(sample_dir, '%d_%d_%d_sample.jpg' % (checkpoint_counter, epoch, test_sum_counter))
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
                checkpoint_counter += 1

                if np.mod(checkpoint_counter, 500) == 2:
                    self.save(checkpoint_counter)

    @property
    def model_dir(self):
        d = "{}_{}_{}_{}".format(
            self.data.name, self.batch_size,
            64, 64)
        return d

    def save(self, step):
        model_name = 'HyperEncoder.model'
        checkpoint_model_dir = os.path.join(self._checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_model_dir):
            os.makedirs(checkpoint_model_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_model_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(' [*] reading checkpoints...')
        checkpoint_model_dir = os.path.join(checkpoint_dir, self.model_dir)

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
