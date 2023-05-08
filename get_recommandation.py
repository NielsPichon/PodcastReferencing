import argparse


def get_recommandation(url_or_path: str):
    raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='URL to the podcast to download '
                        'or path to the podcast audio file.')
    args = parser.parse_args()

    get_recommandation(args.src)
