import configparser
import logging
import os
from argparse import ArgumentParser
from datetime import datetime

import yaml
from box import Box

from breds.breds import BREDS
# from lucene_looper import find_all_text_occurrences
from logging_setup_dla.logging import set_up_root_logger

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

set_up_root_logger(f'BREDS_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.getcwd())

logger = logging.getLogger(__name__)


def main():

    logger.info("Starting BREDS")
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        # cfg = Bothrex(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)


    breads = BREDS(cfg.path.configuration, cfg.path.seeds_file, cfg.path.negative_seeds, cfg.parameters.similarity, cfg.parameters.confidence, cfg.path.objects, cfg.path.vg_objects, cfg.path.vg_objects_anchors)

    if breads.config.coreference:
        cache_paths = cfg.path.coref
    else:
        cache_paths = cfg.path.coref

    htmls_fname = cache_paths.htmls
    tuples_fname = cache_paths.tuples

    breads.generate_tuples(htmls_fname, tuples_fname)
    breads.init_bootstrap(tuples=None)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
