import asyncio
import logging
import pickle

from size_comparisons.scraping import html_scraper
from size_comparisons.scraping.google_ops import create_or_update_results

logger = logging.getLogger(__name__)


def scrape_htmls(html_fname, names: list):

    queries = [[f'{name} length', f'{name} size'] for name in names]
    logger.info(f'Retrieving urls for {len(names)} objects')
    urls = create_or_update_results('urls.pkl', queries, names)

    loop = asyncio.get_event_loop()
    htmls_lookup = html_scraper.create_or_update_urls_html(names, urls, loop)
    with open(html_fname, 'wb') as f:
        pickle.dump(htmls_lookup, f, pickle.HIGHEST_PROTOCOL)
    return htmls_lookup