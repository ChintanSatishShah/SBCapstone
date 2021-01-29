""" Entry point for collector package """
import os.path
import sys
import logging
from logging.handlers import TimedRotatingFileHandler as TRFL
import collector.constants as const
from collector.conf import CollectorConfig as conf
# pylint: disable=invalid-name

if __name__ == '__main__':

    log = logging.getLogger()
    log.setLevel(conf.LOG_LEVEL)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    handler = TRFL(filename=os.path.join(const.APP_ROOT, conf.LOG_DIRECTORY) + const.LOG_FILENAME
                   , when="midnight", interval=1)
    handler.setFormatter(formatter)
    log.addHandler(handler)

    sys.path.insert(0, const.APP_ROOT)
    #sys.path.append(const.ROOT)
    from collector.collect import start_collector

    start_collector()
