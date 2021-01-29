""" Utility methods for Analyzer """
import sys
import os
import logging

from sqlalchemy import create_engine
from sqlalchemy import Table

import analyzer.constants as const

LOG = logging.getLogger(__name__)

def connect_db(db_path, logdb=False):
    """ Validate db connection """
    try:
        engine = create_engine(db_path, echo=logdb)
        engine.connect()
        return engine
    except Exception as err:
        LOG.error("\nConnection to DB %s failed", const.SQLALCHEMY_DATABASE_URI)
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

def create_dir(file_path, purpose=''):
    """ Create directory if it does not exist """
    if not os.path.exists(file_path):
        try:
            os.mkdir(file_path)
            LOG.info("Successfully created the directory %s ", file_path)
        except OSError as err:
            LOG.error("\nCreation of the directory %s failed", file_path)
            LOG.error("\nException Line No.: %s  \nMsg: %s"
                      , str(sys.exc_info()[2].tb_lineno), str(err))
            #raise Exception
    else:
        LOG.info("Output directory%s: %s ", purpose, file_path)
