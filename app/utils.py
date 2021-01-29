""" Utility methods for Analyzer """
import sys
import logging

from sqlalchemy import create_engine
from sqlalchemy import Table

from config import Config as conf

LOG = logging.getLogger(__name__)

def connect_db(logdb=False):
    """ Validate db connection """
    try:
        engine = create_engine(conf.DATABASE_URI, echo=logdb)
        engine.connect()
        return engine
    except Exception as err:
        LOG.error("\nConnection to DB %s failed", conf.DATABASE_URI)
        LOG.error("\nException Line No.: %s  \nMsg: %s"
                  , str(sys.exc_info()[2].tb_lineno), str(err))
        raise Exception

def get_table(table_name, meta):
    """ Validate if table exists """
    try:
        return Table(table_name, meta, autoload=True)
    except Exception as err:
        LOG.error("\nTable %s does not exist", table_name)
        LOG.error("\nException Line No.: %s  \nMsg: %s"
                  , str(sys.exc_info()[2].tb_lineno), str(err))
        raise Exception
