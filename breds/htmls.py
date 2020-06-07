import asyncio
import logging
import pickle
import os

from size_comparisons.scraping import html_scraper
from size_comparisons.scraping.google_ops import create_or_update_results

logger = logging.getLogger(__name__)


def scrape_htmls(html_fname: str, names: list):

    queries = [[f'{name} length', f'{name} size'] for name in names]
    logger.info(f'Retrieving urls for {len(names)} objects')
    urls = create_or_update_results('urls.pkl', queries, names)

    loop = asyncio.get_event_loop()
    htmls_lookup = html_scraper.create_or_update_urls_html(html_fname, names, urls, loop)
    loop.close()
    return htmls_lookup