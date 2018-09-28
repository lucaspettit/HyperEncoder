import tensorflow as tf
import os
from argparse import ArgumentParser
import cv2
import numpy as np
from tqdm import tqdm

try:
    from model import *
    from Data import dataset
except ImportError:
    from hyperencoder.model import *
    from hyperencoder.Data import dataset


parser = ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True,
                    help='path to frozen model')
parser.add_argument('--src', type=str, required=True,
                    help='path to image directory')
parser.add_argument('--dest', type=str, required=True,
                    help='path to destination file. (must be a .CSV)')

args = parser.parse_args()

for p in (args.src, args.model):
    if not os.path.exists(p):
        raise ValueError('%s does not exist!' % p)

model_ext = os.path.splitext(args.model)[-1]
if model_ext != '.pb':
    raise ValueError('%s is not a .protobuf file! [.pb]')

dest_root = os.path.dirname(args.dest)
dest_ext = os.path.splitext(args.dest)[-1]

if dest_ext.lower() != '.csv':
    raise ValueError('destination file is not a CSV!')

if os.path.exists(args.dest):
    raise ValueError('%s is already a file!' % args.dest)

if not os.path.exists(dest_root):
    os.makedirs(dest_root)

# todo: this is special formatting for my GIPHY dataset
files = os.listdir(args.src)
flicks = {}
for f in files:
    flick_id, person_id, frame_id = [int(elem) for elem in os.path.splitext(f)[0].split('-')]

    if flick_id not in flicks:
        flicks[flick_id] = {}

    if person_id not in flicks[flick_id]:
        flicks[flick_id][person_id] = []

    flicks[flick_id][person_id].append(frame_id)

files = []
for flick_id, flick in flicks.items():
    for person_id, person in flick.items():

        person = sorted(person)
        fnames = [(flick_id, person_id, frame_id) for frame_id in person]
        files += fnames

# clear memory!
del flicks

# todo: now we got a sorted list of files :)
with tf.Session() as sess:

    encoder = HyperEncoder.load_frozen(sess, args.model)

    i = 0
    N = len(files)
    batch_size = encoder.batch_size

    print('starting conversion...')
    with open(args.dest, 'w') as outfile:

        header = 'flick_id,person_id,frame_id,'
        header += (','.join(['x%d' % i for i in range(encoder.emb_dim)]))
        header += '\n'
        outfile.write(header)

        counter = 0
        counter_tick = 1000
        print('%d total files to process' % N)

        while i < N:

            if counter >= counter_tick:
                print('%d files processed [%.4f]' % (counter, float(counter) / float(N)))
                counter_tick = min(counter_tick + 1000, N - 1)
            counter += 1

            end = min(N, i + batch_size)
            fs = files[i: end]
            batch = []
            for f in fs:
                f = os.path.join(args.src, '%d-%d-%d.jpg' % (f[0], f[1], f[2]))
                img = cv2.imread(f)
                img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
                if len(img.shape) < 3 or img.shape[2] == 1:
                    h, w = img.shape[:2]
                    _img = np.zeros(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
                    for i in range(3):
                        _img[:, :, i] = img
                    img = _img
                batch.append(img)

            size = end - i
            batch += [np.zeros((227, 227, 3), dtype=np.uint8) for _ in range(batch_size - size)]

            embs = encoder.encode(batch)

            values = ''
            for i in range(size):
                e = embs[i]
                f = fs[i]
                values += '%d,%d,%d,' % f
                values += (','.join([str(value) for value in e]))
                values += '\n'

            outfile.write(values[:-1])