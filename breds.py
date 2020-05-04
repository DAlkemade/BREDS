import configparser
import logging
import os
from argparse import ArgumentParser
from datetime import datetime

from breds.breds import BREDS
# from lucene_looper import find_all_text_occurrences
from logging_setup_dla.logging import set_up_root_logger

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

set_up_root_logger(f'BREDS_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.getcwd())

logger = logging.getLogger(__name__)


def main():

    logger.info("Starting BREDS")
    parser = ArgumentParser()
    parser.add_argument('--configuration', type=str, required=True)
    parser.add_argument('--seeds_file', type=str, required=True)
    parser.add_argument('--negative_seeds', type=str, required=True)
    parser.add_argument('--similarity', type=float, required=True)
    parser.add_argument('--confidence', type=float, required=True)
    parser.add_argument('--objects', type=str, required=True)
    parser.add_argument('--cache_config_fname', type=str, required=True)
    parser.add_argument('--vg_objects', type=str, required=True)
    parser.add_argument('--vg_objects_anchors', type=str, required=True)
    args = parser.parse_args()

    breads = BREDS(args.configuration, args.seeds_file, args.negative_seeds, args.similarity, args.confidence, args.objects, args.vg_objects, args.vg_objects_anchors)

    cache_config = configparser.ConfigParser()
    cache_config.read(args.cache_config_fname)
    cache_type = 'COREF' if breads.config.coreference else 'NOCOREF'
    htmls_fname = cache_config[cache_type].get('htmls')
    tuples_fname = cache_config[cache_type].get('tuples')

    breads.generate_tuples(htmls_fname, tuples_fname)
    breads.init_bootstrap(tuples=None)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
