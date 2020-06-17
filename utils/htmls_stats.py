import logging
import os
import pickle

from logging_setup_dla.logging import set_up_root_logger
from nltk import tokenize
import numpy as np

set_up_root_logger(f'HTMLSSTATS', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)


def main():

    with open('htmls.pkl', "rb") as f_html:
        results: dict = pickle.load(f_html)

    sizes = []
    for htmls in results.values():
        for html in htmls:
            words = tokenize.word_tokenize(html)
            sizes.append(len(words))

    logger.info(f'Mean doc size: {np.mean(sizes)}')
    logger.info(f'Median doc size: {np.median(sizes)}')



if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
