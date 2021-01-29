"""Constants"""
import os.path
from datetime import datetime, date

#import os
ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../')
APP_ROOT = r''
LOG_FILENAME = 'collector.log'
SQLALCHEMY_DATABASE_URI = r'sqlite:///' + APP_ROOT + r'./db/StockSeasonality.db'

NSE = 'NSE'
LISTING_TYPE = 'Equity'
SYSTEM = 'System'
MAX_FETCH_ATTEMPTS = 5
API_SUFFIX = '.NS'
FINANCE_API = 'yahoo'

#Initialise the start date from which the historical stock data is to be fetched
START_DATE = date(1990, 1, 1)
TODAY = date.today()
CURRENT_YEAR = datetime.today().year
