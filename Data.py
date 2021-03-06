import os
import numpy as np
from queue import Queue
import threading
import cv2
from random import shuffle as _shuffle, randint
import time


class dataset(object):
    """
    Dataset class.
    mulithreaded read images from folder + create batches
    """

    def __init__(self,
                 train_dir,
                 batch_size,
                 test_dir = None,
                 x_shape=(227, 227),
                 y_shape=None,
                 split=0.9,
                 name=None,
                 keep_grayscale=True,
                 shuffle=True):



        if name is None:
            self.name = os.path.basename(train_dir)
        else:
            self.name = name

        self._x_shape = tuple(x_shape)
        self._y_shape = x_shape if y_shape is None else tuple(y_shape)
        self._batch_size = batch_size

        # check if x is 2D
        self._input_2d = len(x_shape) == 2 or x_shape[2] == 1

        self._keep_grayscale = keep_grayscale


        # if we're only given one dataset path (i.e. train_dir), then we need to split the dataset
        if test_dir is None:
            datapaths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]

            if shuffle:
                _shuffle(datapaths)

            split_index = int(len(datapaths) * split)
            self._training_paths = datapaths[:split_index]
            self._test_paths = datapaths[split_index:]
            self.num_batches = int(len(self._training_paths) / batch_size)

        # if we're given two dataset paths (i.e. train_dir + test_dir) then we do NOT need to split the dataset
        else:
            self._training_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
            self._test_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]

            if shuffle:
                _shuffle(self._training_paths)
                _shuffle(self._test_paths)

            self.num_batches = int(len(self._training_paths) / batch_size)

        self._num_training_data = len(self._training_paths)
        self._num_test_data = len(self._test_paths)

        self._next_training_index = 0
        self._next_test_index = 0

        self._training_batch_q = Queue(maxsize=5)
        self._test_batch_q = Queue(maxsize=3)

        self._batch_training_thread = threading.Thread(target=self._fill_training_batch_q,
                                              name='t-dataset-fillTrainingQueue')
        self._batch_training_thread.start()

        self._batch_test_thread = threading.Thread(target=self._fill_test_batch_q,
                                                   name='t-dataset-fillTestQueue')
        self._batch_test_thread.start()

    def _get_next_filepath(self, training=False):
        filename = ''
        if training:
            if self._next_training_index >= self._num_training_data:
                self._next_training_index = 0

            filename = self._training_paths[self._next_training_index]
            self._next_training_index += 1
        else:
            if self._next_test_index >= self._num_test_data:
                self._next_test_index = 0

            filename = self._test_paths[self._next_test_index]
            self._next_test_index += 1
        return filename
    
    @staticmethod
    def normalize_image(img):
        norm_img = img / (255.0 / 2.0)
        norm_img = norm_img - 1.0
        return norm_img

    @staticmethod
    def denormalize_image(norm_img):
       img = norm_img + 1.0
       img = img * (255.0 / 2.0)
       return img

    @staticmethod
    def _grayscaleToRGB(gray):
        h, w = gray.shape[:2]
        img = np.zeros(h * w * 3,dtype=np.uint8).reshape(h, w, 3)
        for i in range(3):
            img[:,:,i] = gray
        return img

    def _get_xy(self, filename, keep_grayscale):
        res = None
        try:
            img = cv2.imread(filename)

            # check for and handle grayscale images
            shape = img.shape
            if len(shape) < 3 or shape[2] == 1:
                if not keep_grayscale:
                    return None
                elif not self._input_2d:
                    img = self._grayscaleToRGB(img)

            y = cv2.resize(img, self._y_shape[:2], interpolation=cv2.INTER_CUBIC)
            x = cv2.resize(img, self._x_shape[:2], interpolation=cv2.INTER_CUBIC)

            if self._input_2d:
                x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
                x = x.reshape(self._x_shape)

            # normalize y to be within tanh values [-1, 1]
            y = self.normalize_image(y)
            res = x, y

        except Exception as e:
            print('  ERROR !!\n%s' % str(e))
            pass
        return res

    def _fill_training_batch_q(self):
        t_name = threading.current_thread().getName()
        #print('%s: starting' % t_name)

        while self._training_batch_q.not_full:
            batch_x = []
            batch_y = []

            #print('%s: batch queue not full. filling queue.' % t_name)

            while len(batch_x) < self._batch_size:
                filename = self._get_next_filepath(training=True)

                res = self._get_xy(filename, self._keep_grayscale)
                if res is None:
                    continue
                x, y = res
                batch_x.append(x)
                batch_y.append(y)

            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            self._training_batch_q.put((batch_x, batch_y))

        #print('%s: exiting' % t_name)

    def _fill_test_batch_q(self):

        t_name = threading.current_thread().getName()
        #print('%s: starting' % t_name)

        while self._test_batch_q.not_full:
            batch_x, batch_y = [], []
            #print('%s: batch queue not full. filling queue.' % t_name)

            while len(batch_x) < self._batch_size:
                filename = self._get_next_filepath(training=False)
                res = self._get_xy(filename, True)
                if res is None:
                    continue
                x, y = res
                batch_x.append(x)
                batch_y.append(y)

            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            self._test_batch_q.put((batch_x, batch_y))

    def next_batch(self, training=False):
        batch = None, None
        if training:
            while self._training_batch_q.empty():
                #print('t-main: training batch queue empty. sleeping for 1 second')
                time.sleep(1)
                pass
            batch = self._training_batch_q.get()
        else:
            while self._test_batch_q.empty():
                #print('t-main: testing batch queue empty. sleeping for 1 second')
                time.sleep(1)
                pass
            batch = self._test_batch_q.get()
        return batch

    def random_datapoint(self, training=False):
        while True:
            if training:
                idx = randint(0, self._num_training_data - 1)
                filename = self._training_paths[idx]
            else:
                idx = randint(0, self._num_test_data - 1)
                filename = self._test_paths[idx]

            res = self._get_xy(filename, self._keep_grayscale)
            if res is None:
                continue

            return res
