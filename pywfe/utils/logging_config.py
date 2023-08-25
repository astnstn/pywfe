import logging


def log(level=logging.INFO):

    # Create a logger with the name 'pywfe'
    logger = logging.getLogger('pywfe')

    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(level)

    # Create a console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # Create a formatter with your specified format and date format
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                                  datefmt='%Y-%m-%d:%H:%M:%S')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
