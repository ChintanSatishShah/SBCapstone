#""" Main file to initiate app """
"""Initializes application creating a Flask app instance"""
import os.path
import sys
import logging
from logging.handlers import TimedRotatingFileHandler as TRFL

from flask import Flask, Blueprint
from flask import current_app as app
from flask.logging import create_logger
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy import and_, or_

from .utils import connect_db, get_table

# pylint: disable=invalid-name

#ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../')
APP_ROOT = r''
CONFIG_ROOT = sys.path.append(os.path.abspath(os.path.join('..', 'config')))

#sys.path.insert(0, APP_ROOT)
#sys.path.append(ROOT)
sys.path.append(CONFIG_ROOT)
#bp = Blueprint('results', __name__, static_url_path='/static/results', static_folder=OUTPUT_PATH)
bp = Blueprint('results', __name__, static_url_path='/static/results')
dbase = SQLAlchemy()

#Check DB connection
db = connect_db(False)
metadata = MetaData(db)
#Create Table object
tracker = get_table('ForecastTracker', metadata)

def create_app():
    """Initialize the core application."""
    appln = Flask(__name__, instance_relative_config=False)
    #appln.config.from_pyfile('../config.py')
    appln.config.from_object('config.Config')
    #LOG = create_logger(appln)

    # Initialize Plugins
    dbase.init_app(appln)

    log = logging.getLogger(__name__)
    log.setLevel(appln.config.get("LOG_LEVEL"))
    formatter = logging.Formatter(appln.config.get("LOG_FORMAT"))
    handler = TRFL(filename=os.path.join(APP_ROOT, appln.config.get("LOG_DIRECTORY")) \
                   + 'app.log' #const.LOG_FILENAME
                   , when="midnight", interval=1)
    handler.setFormatter(formatter)
    log.addHandler(handler)

    with appln.app_context():
        # Include our Routes
        from . import views # pylint: disable=unused-import
        #from .models import Stock, Execution

        # Register Blueprints
        appln.register_blueprint(bp)
        #app.register_blueprint(admin.admin_bp)

        return appln
