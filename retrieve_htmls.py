import asyncio
import pickle
from argparse import ArgumentParser
from pathlib import Path

from size_comparisons.scraping import html_scraper

from breds.config import read_objects_of_interest


def main():
    parser = ArgumentParser()
    parser.add_argument('--htmls_fname', type=str, required=True)
    parser.add_argument('--objects_fname', type=str, required=True)
    args = parser.parse_args()
    html_fname: str = args.htmls_fname
    objects_path = Path(args.objects_fname)
    with open('urls.pkl', 'rb') as f:
        urls = pickle.load(f)

    names = list(read_objects_of_interest(objects_path))
    print(f'Number of objects: {len(names)}')

    loop = asyncio.get_event_loop()
    htmls_lookup = html_scraper.create_or_update_urls_html(names, urls, loop)
    with open(html_fname, 'wb') as f:
        pickle.dump(htmls_lookup, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()