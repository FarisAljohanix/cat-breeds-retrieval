# setup logging
import logging

def setup_logging(name: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    httpx = logging.getLogger("httpx")
    httpx.setLevel(logging.WARNING) # to prvent annoying logs from httpx
    return logger