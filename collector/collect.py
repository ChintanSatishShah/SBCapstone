""" Fetch and store stock data """
import sys
import os.path
import logging

from datetime import datetime
#import time
import pandas as pd

#Used to fetch NSE stock list with tickers
from nsetools import Nse

#Used to fetch NSE stock historical data
import pandas_datareader as pdr
from pandas_datareader._utils import RemoteDataError
from requests import ReadTimeout

from sqlalchemy import create_engine, MetaData
from sqlalchemy import Table

import collector.constants as const
from collector.conf import CollectorConfig as conf

# pylint: disable=invalid-name
#Instantiate global variables
log = logging.getLogger(__name__)
DATA_PATH = os.path.join(const.APP_ROOT, conf.RAWDATA_DIRECTORY)

def connect_db(db_path, logdb=False):
    """ Validate db connection """
    try:
        engine = create_engine(db_path, echo=logdb)
        engine.connect()
        return engine
    except Exception as err:
        log.error("\nConnection to DB %s failed", const.SQLALCHEMY_DATABASE_URI)
        log.error("\nException Line No.: %s  \nMsg: %s"
                  , str(sys.exc_info()[2].tb_lineno), str(err))
        raise Exception

def get_table(table_name, meta):
    """ Validate if table exists """
    try:
        return Table(table_name, meta, autoload=True)
    except Exception as err:
        log.error("\nTable %s does not exist", table_name)
        log.error("\nException Line No.: %s  \nMsg: %s"
                  , str(sys.exc_info()[2].tb_lineno), str(err))
        raise Exception

def create_dir(file_path):
    """ Create directory if it does not exist """
    if not os.path.exists(file_path):
        try:
            os.mkdir(file_path)
            log.info("Successfully created the directory %s ", file_path)
        except OSError as err:
            log.error("\nCreation of the directory %s failed", file_path)
            log.error("\nException Line No.: %s  \nMsg: %s"
                      , str(sys.exc_info()[2].tb_lineno), str(err))
            #raise Exception
    else:
        log.info("Output directory for Raw Data Collector: %s ", file_path)

def get_stock_list(isTest=False):
    """
    Get the List of stocks listed on NSE
    Save list in file and create entries in DB
    Returns the count of stocks listed on NSE
    """

    log.info("Enter - Get stock list")

    #Fetch stock list using Python NSE tools library
    stock_codes = Nse().get_stock_codes()
    log.info("No. of stocks listed on NSE: %d ", len(stock_codes) - 1)

    #Save stock list as file
    pd.DataFrame(stock_codes.items()).to_csv(DATA_PATH + 'NSE_StockList.csv'
                                             , header=False, index=True)

    #Remove header row
    stock_codes.pop(next(iter(stock_codes)))

    i = tracker.insert()
    if not isTest:
        #Insert a row for NIFTY Index
        i.execute({'ForecastYear': const.CURRENT_YEAR,\
                   'ListingIndex': const.NSE,\
                   'ListingName': "NIFTY 50",\
                   'ListingTicker': "^NSEI",\
                   'ListingType': "INDEX",\
                   #Set CrossValidationCount = -1 to avoid further analysis
                   'CrossValidationCount': -1,\
                   'CreatedDate': datetime.now(),\
                   'CreatedBy': const.SYSTEM\
                   })
    #Save stock list in DB
    for ticker, name in stock_codes.items():
        #print(f"Saving row for: {name} with ticker: {ticker}")
        log.info("Saving row for: %s with ticker: %s", name, ticker)
        if isTest:
            break
        else:
            i.execute({'ForecastYear': const.CURRENT_YEAR,\
                       'ListingIndex': const.NSE,\
                       'ListingName': name,\
                       'ListingTicker': ticker,\
                       'ListingType': const.LISTING_TYPE,\
                       'CreatedDate': datetime.now(),\
                       'CreatedBy': const.SYSTEM\
                       })

    log.info("Exit - Get stock list")
    #Return the count of stocks listed on NSE
    return len(stock_codes)

def get_pending_stocks(throttle=2000):
    """ Get the list of stocks for which historical data is not loaded (based on status) """
    squery = tracker.select((tracker.c.ForecastYear == const.CURRENT_YEAR)
                            & (tracker.c.RawDataFilePath == None) #pylint: disable=singleton-comparison
                            & (tracker.c.DataCollectionAttempt < const.MAX_FETCH_ATTEMPTS)
                            ).limit(throttle)
    return squery.execute().fetchall()

def get_history_data(lstocks, isTest=0):
    """
    Fetch the historical price data
    Save the historical price data in csv format
    Update the database to mark completion of fetching historical
    """

    incomplete = ""
    complete = ""
    error = ""

    for row in lstocks:
        log.info("Fetching data for Stock: %s with ticker: %s"
                 , row.ListingName, row.ListingTicker)
        try:
            if isTest == 2:
                #RemoteDataError simulation
                data = pdr.DataReader(row.ListingTicker, const.FINANCE_API
                                      , start=const.START_DATE, end=const.TODAY)
            elif isTest == 3:
                #ValueError simulation
                data = pdr.DataReader(row.ListingTicker, const.FINANCE_API
                                      , start=const.TODAY, end=const.START_DATE)
            else:
                if row.ListingType.upper() != "INDEX":
                    data = pdr.DataReader(row.ListingTicker + const.API_SUFFIX, const.FINANCE_API
                                          , start=const.START_DATE, end=const.TODAY)
                else:
                    data = pdr.DataReader(row.ListingTicker, const.FINANCE_API
                                          , start=const.START_DATE, end=const.TODAY)
                #print(f"Data fetched successfully for Stock: {row.ListingName}")
                log.info("Data fetched successfully for Stock: %s ", row.ListingName)
        except RemoteDataError as rderr:
            log.error("Finance API could not fetch data for Stock: %s", row.ListingName)
            log.error("\nException Line No.: %s  \nMsg: %s"
                      , str(sys.exc_info()[2].tb_lineno), str(rderr))
            if isTest < 1:
                tracker.update().\
                        where(tracker.c.ForecastID == row.ForecastID).\
                        values(DataCollectionAttempt=row.DataCollectionAttempt + 1).\
                        execute()
            error += '\n' + row.ListingName
        except ReadTimeout as rtoerr:
            log.error("API read timeout while fetching data for Stock: %s", row.ListingName)
            log.error("\nException Line No.: %s  \nMsg: %s"
                      , str(sys.exc_info()[2].tb_lineno), str(rtoerr))
            error += '\n' + row.ListingName
        except ConnectionError as connerr:
            log.error("API read timeout while fetching data for Stock: %s", row.ListingName)
            log.error("\nException Line No.: %s  \nMsg: %s"
                      , str(sys.exc_info()[2].tb_lineno), str(connerr))
            error += '\n' + row.ListingName
        except ValueError as err:
            log.error("Failure in fetching data for Stock: %s", row.ListingName)
            log.error("\nException Line No.: %s  \nMsg: %s"
                      , str(sys.exc_info()[2].tb_lineno), str(err))
            error += '\n' + row.ListingName
            #raise Exception
        else:
            #If only 1 row is retrieved, it indicates an error in API
            if len(data) > 1:
                raw_path = DATA_PATH + row.ListingTicker + '.csv'
                data.to_csv(raw_path, header=True, index=True)
                #print(f"Data saved for Stock: {row.ListingName}")
                log.info("Data saved for Stock: %s", row.ListingName)
                if isTest < 1:
                    #Update status in DB
                    tracker.update().\
                        where(tracker.c.ForecastID == row.ForecastID).\
                        values(RawDataFilePath=raw_path
                               , AvailableFrom=data.index[0].strftime('%Y-%m-%d %H:%M:%S')
                               , DataCollectedAt=datetime.now()
                               , DataCollectionAttempt=row.DataCollectionAttempt + 1).\
                    execute()
                complete += '\n' + row.ListingName
            else:
                #print(f"Complete data not received for Stock: {row.ListingName}")
                log.info("Complete data not received for Stock: %s", row.ListingName)
                if isTest < 1:
                    tracker.update().\
                        where(tracker.c.ForecastID == row.ForecastID).\
                        values(DataCollectedAt=datetime.now()
                               , DataCollectionAttempt=const.MAX_FETCH_ATTEMPTS).\
                        execute()
                incomplete += '\n' + row.ListingName

    if error:
        log.info("Error in fetching data for following stocks: %s", error)
    if incomplete:
        log.info("Complete data not received for following stocks: %s", incomplete)
    if complete:
        log.info("Historical price data saved successfully for following stocks: %s", complete)
    log.info("Completed a cycle of fetching data for stocks")

#Create output directory to store raw stock data
create_dir(DATA_PATH)

#Check DB connection
db = connect_db(const.SQLALCHEMY_DATABASE_URI, False)
metadata = MetaData(db)

#Create Table object
tracker = get_table('ForecastTracker', metadata)

def start_collector(isTest=False):
    """ Collector Code """
    log.info("Start collector")

    #Check if data alreay exists for current year
    #Ignore some rows added for unit testing - 10
    cquery = tracker.count((tracker.c.ForecastYear == const.CURRENT_YEAR)
                           & (~tracker.c.ListingTicker.like('%TEST%'))
                           )
    stock_cnt = cquery.execute().fetchone()[0]

    if (stock_cnt == 0) | isTest:
        #If data does not exist
        stock_cnt = get_stock_list(isTest)

    retry = True
    while retry:
        #Throttling no. of stocks per cycle, typically 3 or 5
        limit = conf.STOCKS_PER_CYCLE
        if isTest:
            #Get limited stocks in case of Unit test
            limit = 3

        pending_stocks = get_pending_stocks(limit)
        pending_stock_cnt = len(pending_stocks)
        log.info("Raw data collection started for %d out of total %d stocks."
                 , pending_stock_cnt, stock_cnt)

        if (pending_stock_cnt == 0) | isTest:
            #Skip in case of unit testing
            #End while loop after there are no more stocks pending
            retry = False
        else:
            #Fetch historical data for remaining stocks
            get_history_data(pending_stocks)

    #Exit code
    log.info("All possible stocks data is fetched after trying upto %d times"
             , const.MAX_FETCH_ATTEMPTS)
    query = tracker.count((tracker.c.ForecastYear == const.CURRENT_YEAR)
                          & (tracker.c.RawDataFilePath == None) #pylint: disable=singleton-comparison
                         )
    err_stock_cnt = query.execute().fetchone()[0]
    log.info("Raw data is not available for %d out of total %d stocks."
             , err_stock_cnt, stock_cnt)
    log.info("Exiting collector")
