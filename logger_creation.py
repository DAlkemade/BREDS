import logging
import sys
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "my_app.log"


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger

def set_unhandled_exceptions_catch(logger: logging.Logger):
    def my_handler(exc_type, exc_value, exc_traceback):
        """Handler for unhandled exceptions that will write to the logs"""
        if issubclass(exc_type, KeyboardInterrupt):
            # call the default excepthook saved at __excepthook__
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = my_handler

