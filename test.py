from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--x_dim', nargs='+', type=int, default=[10, 20, 30], help='add ints')
args = parser.parse_args()

print(args)
