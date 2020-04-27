from argparse import ArgumentParser
from pathlib import Path

from breds.htmls import scrape_htmls
from parse_coref import get_all_objects


def main():
    parser = ArgumentParser()
    parser.add_argument('--htmls_fname', type=str, required=True)
    parser.add_argument('--objects_fname', type=str, required=True)
    args = parser.parse_args()
    html_fname: str = args.htmls_fname
    objects_path = Path(args.objects_fname)
    names = get_all_objects(objects_path)
    print(f'Number of objects: {len(names)}')

    scrape_htmls(html_fname, names)


if __name__ == "__main__":
    main()