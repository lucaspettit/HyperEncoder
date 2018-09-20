import os
import numpy as np
from queue import Queue
import threading
import cv2
from random import _shuffle, randint


class dataset(object):
    """
    Dataset class.
    mulithreaded read images from folder + create batches
    """

    def __init__(self, input_dir, batch_size, x_shape, y_shape=None, split=0.9, name=None, shuffle=True):
        datapaths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                     if os.path.isfile(os.path.join(input_dir, f))]

        if shuffle:
            _shuffle(datapaths)

        if name is None:
            self.name = os.path.basename(input_dir)
        else:
            self.name = name

        self._x_shape = x_shape
        self._y_shape = x_shape if y_shape is None else y_shape
        self._batch_size = batch_size
        self.num_batches = int(len(datapaths) / batch_size)

        split_index = int(len(datapaths) * split)
        self._training_paths = datapaths[:split_index]
        self._test_paths = datapaths[split_index:]

        self._num_training_data = len(self._training_paths)
        self._num_test_data = len(self._test_paths)

        self._next_training_index = 0
        self._next_test_index = 0

        self._training_batch_q = Queue(maxsize=3)
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

    def _get_xy(self, filename):
        res = None
        try:
            img = cv2.imread(filename)
            # check for and handle grayscale images
            shape = img.shape
            if len(shape) < 3 or len(shape[2]) == 1:
                h, w = shape
                img2 = np.zeros(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
                for i in range(3):
                    img2[:, :, i] = img[:, :]
                img = img2

            y = cv2.resize(img, self._y_shape, interpolation=cv2.INTER_CUBIC)
            x = cv2.resize(img, self._x_shape, interpolation=cv2.INTER_CUBIC)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            # normalize y to be within tanh values [-1, 1]
            y = self.normalize_image(y)
            res = x, y

        except Exception as e:
            pass
        return res

    def _fill_training_batch_q(self):
        while self._training_batch_q.not_full:
            batch_x = []
            batch_y = []

            while len(batch_x) < self._batch_size:
                filename = self._get_next_filepath(training=True)

                res = self._get_xy(filename)
                if res is None:
                    continue
                x, y = res
                batch_x.append(x)
                batch_y.append(y)

            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            self._training_batch_q.put((batch_x, batch_y))

    def _fill_test_batch_q(self):
        while self._test_batch_q.not_full:
            batch_x, batch_y = [], []

            while len(batch_x) < self._batch_size:
                filename = self._get_next_filepath(training=False)
                res = self._get_xy(filename)
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
                pass
            batch = self._training_batch_q.get()
        else:
            while self._test_batch_q.empty():
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

            res = self._get_xy(filename)
            if res is None:
                continue

            return res
