import logging

def setup_logger(log_file):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode='w')
    log = logging.getLogger(__name__)
    return log