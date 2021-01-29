""" Entry point for collector package """
import os.path
import logging
from logging.handlers import TimedRotatingFileHandler as TRFL
import analyzer.constants as const
from analyzer.conf import AnalyzerConfig as conf
# pylint: disable=invalid-name

if __name__ == '__main__':

    log = logging.getLogger()
    log.setLevel(conf.LOG_LEVEL)
    formatter = logging.Formatter(conf.LOG_FORMAT)
    handler = TRFL(filename=os.path.join(const.APP_ROOT, conf.LOG_DIRECTORY) + const.LOG_FILENAME
                   , when="midnight", interval=1)
    handler.setFormatter(formatter)
    log.addHandler(handler)

    #sys.path.insert(0, const.APP_ROOT)
    #sys.path.append(const.ROOT)
    from analyzer.analyze import start_analysis

    start_analysis()
