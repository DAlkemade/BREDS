import asyncio
import pickle
from argparse import ArgumentParser
from pathlib import Path

from size_comparisons.scraping import html_scraper
from size_comparisons.scraping.google_ops import create_or_update_results

from breds.config import read_objects_of_interest
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


def get_objects(objects_path):

    return list(read_objects_of_interest(objects_path))


def scrape_htmls(html_fname, names):

    queries = [[f'{name} length', f'{name} size'] for name in names]
    urls = create_or_update_results('urls.pkl', queries, names)

    loop = asyncio.get_event_loop()
    htmls_lookup = html_scraper.create_or_update_urls_html(names, urls, loop)
    with open(html_fname, 'wb') as f:
        pickle.dump(htmls_lookup, f, pickle.HIGHEST_PROTOCOL)
    return htmls_lookup


if __name__ == "__main__":
    main()