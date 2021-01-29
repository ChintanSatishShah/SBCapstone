"""This is where you define models for your application"""
import logging
from sqlalchemy import func

from . import dbase, tracker
from .constants import CURRENT_YEAR

# pylint: disable=invalid-name
LOG = logging.getLogger(__name__)

def get_stock_list():
    """ Get the complete list of stocks for Current Year """
    #res = Stock.query.all()
    #res = Stock.query.filter(Stock.ListingName.startswith('T')).all()
    #return Stock.query.filter(Stock.ForecastYear == CURRENT_YEAR).all()
    return Stock.query.\
        filter((func.upper(Stock.ListingType) != "INDEX") & (Stock.ForecastYear == CURRENT_YEAR))\
        .all()

def get_forecast_status(ticker):
    """ Get the Forecast status of selected Stock """
    #fc_row = Stock.query.\
        #filter((Stock.ListingTicker == ticker) & (Stock.ForecastYear == CURRENT_YEAR))\
        #.order_by(Stock.ForecastID.desc(), Stock.ForecastYear.desc())\
        #.all()
    #exec_row = Execution.query.\
        #filter(Execution.ForecastID == fc_row[0].ForecastID).\
        #all()
    #if exec_row:
        #print(exec_row[0].ExecutionStepID)
    #print('No exeuction data available for stock:', fc_row[0].ListingName)
    return tracker.select((tracker.c.ForecastYear == CURRENT_YEAR)
                          & (tracker.c.ListingTicker == ticker)
                          ).execute().fetchone()

class Stock(dbase.Model):
    """ Stock class """
    __tablename__ = 'ForecastTracker'

    #Table columns
    ForecastID = dbase.Column(dbase.Integer, primary_key=True)
    ForecastYear = dbase.Column(dbase.Integer, nullable=False)
    ListingIndex = dbase.Column(dbase.String(20), unique=False, nullable=True)
    ListingTicker = dbase.Column(dbase.String(20), unique=False, nullable=False)
    ListingName = dbase.Column(dbase.String(100), unique=False, nullable=False)
    ListingType = dbase.Column(dbase.String(20), unique=False, nullable=False)

    #def __repr__(self):
        #return '<Stock {}>'.format(self.ListingName)

    def as_dict(self):
        """ Return as dict object from class """
        #Vaidate using http://127.0.0.1:5000/stocks
        return {'stockName': self.ListingName}
        #return {'stockName': self.ListingName, 'stockTicker': self.ListingTicker}

class Execution(dbase.Model):
    """ Execution class """
    __tablename__ = 'ExecutionStep'

    #Table columns
    ExecutionStepID = dbase.Column(dbase.Integer, primary_key=True)
    ForecastID = dbase.Column(dbase.Integer, nullable=False)
    StepCode = dbase.Column(dbase.Integer, nullable=False)
