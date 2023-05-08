import argparse


def index_podcasts(num: int):
    raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num', type=int, help='Number of new podcasts to index')
    args = parser.parse_args()

    index_podcasts(args.num)
