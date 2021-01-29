""" Analyze stock data for Seasonality """
import logging
from datetime import datetime
import sys

#from datetime import datetime
#import pandas as pd
#import matplotlib.pyplot as plt

from sqlalchemy import MetaData
#from sqlalchemy.sql.functions import func
from sqlalchemy import asc, desc

from analyzer.utils import connect_db, get_table, create_dir
import analyzer.constants as const
from analyzer.constants import wd_col, MAX_FETCH_ATTEMPTS
from analyzer.transform import get_rawdata, transform_data, agg_data
from analyzer.forecast import crossvalidate, predict#, result_row
from analyzer.plot import plot_data

#Other project dependencies
#from collector.constants import MAX_FETCH_ATTEMPTS

LOG = logging.getLogger(__name__)

def get_tba_stocks(size=10000):
    """ Get the list of stocks for which tranformation is pending """
    if size == 1:
        #Fetch only 1 stock at a time
        squery = tracker.select((tracker.c.ForecastYear == const.CURRENT_YEAR)
                                & (tracker.c.RawDataFilePath != None) #pylint: disable=singleton-comparison
                                & (tracker.c.CrossValidationCount == 0)
                                #order by oldest date since data is Available
                                ).order_by(asc(tracker.c.AvailableFrom))\
                                .limit(size)#Limit selected rows
        return squery.execute().fetchall()
    else:
        return tracker.count((tracker.c.ForecastYear == const.CURRENT_YEAR)
                             & (tracker.c.RawDataFilePath != None) #pylint: disable=singleton-comparison
                             & (tracker.c.CrossValidationCount == 0)
                             ).execute().fetchone()[0]

def transform(row):
    """ Data tranformation and aggregation for selected stock """

    ticker = row.ListingTicker
    name = row.ListingName

    LOG.info("Starting data transformation for Stock: %s with ticker: %s", name, ticker)

    data = transform_data(get_rawdata(const.RAWDATA_PATH + ticker + '.csv'))
    if data.empty:
        LOG.info("Less than 1 year of historical stock data is available for %s", name)
        #Update status in DB
        tracker.update().\
            where(tracker.c.ForecastID == row.ForecastID).\
            values(CrossValidationCount=-1 #To differentiate from un-cleaned stocks
                   , LastModifiedDate=datetime.now()).\
            execute()
    else:
        wd_series = data[wd_col]
        #pctile_series = data[pctile_col]

        year_agg_data = agg_data(data, wd_series)
        #(year_agg_data.index.values)

        #Remove years when Adjusted close price < 0
        data = data[data.year >= year_agg_data.index.min()]

        #Validate if sufficient data is available for forecasting
        years_of_data = len(year_agg_data) #len(data.index.year.unique())
        #print("Historical stocks data for", name, "is available for", years_of_data, "years")
        LOG.info("Historical stocks data for %s is available for %d years", name, years_of_data)

        plot_data(ticker, data, year_agg_data)

        #Save data
        mod_path = const.MODDATA_PATH + ticker + '.csv'
        data.to_csv(mod_path, header=True, index=True)
        year_agg_data.to_csv(mod_path.replace('.csv', const.FILE_YR_AGG), header=True, index=True)
        #print(f"Transformed and Aggregated Data saved for Stock: {name}")
        LOG.info("Transformed and Aggregated Data saved for Stock: %s", name)

        cross_val = -1
        # Check min. years of data required for forecasting
        if years_of_data < const.min_years_data_reqd:
            #print("Historical stocks data for", name, "is not sufficient for Time series analysis")
            LOG.info("Historical stock data for %s is insufficient for Time series analysis", name)
        else:
            cross_val = min(years_of_data - const.min_trn_yrs, const.max_cross_val)
            #print('No. of cross validations, Test data period:', cross_val)
            LOG.info("No. of cross validations, Test data period: %d years", cross_val)

        #Update status in DB
        tracker.update().\
            where(tracker.c.ForecastID == row.ForecastID).\
            values(InputDataFilePath=mod_path
                   , CrossValidationCount=cross_val
                   , DataAnalyzedFrom=data.index[0].strftime('%Y-%m-%d %H:%M:%S')
                   , LastModifiedDate=datetime.now()).\
            execute()

    LOG.info("Completed data transformation for Stock: %s with ticker: %s", name, ticker)

def get_ut_stock():
    """ Get one fixed row for unit testing """
    squery = tracker.select(tracker.c.ListingTicker == 'UNITTEST')
    return squery.execute().fetchall()

def get_tsa_pending_stocks(size=10000):
    """ Get the list of stocks for which Time Series analysis is pending """
    if size == 1:
        #Fetch only 1 stock at a time
        squery = tracker.select((tracker.c.ForecastYear == const.CURRENT_YEAR)
                                #& (tracker.c.ListingTicker == 'A2ZINFRA')
                                & (tracker.c.CrossValidationCount > 0)
                                & (tracker.c.SeasonalityPredicted == None) #pylint: disable=singleton-comparison
                                & (tracker.c.AnalysisAttempt < const.MAX_ANALYZE_ATTEMPTS)
                                #order by latest date since data is Available
                                ).order_by(asc(tracker.c.AnalysisAttempt)
                                           , desc(tracker.c.AvailableFrom))\
                                .limit(size)#Limit selected rows
        return squery.execute().fetchall()
    else:
        return tracker.count((tracker.c.ForecastYear == const.CURRENT_YEAR)
                             & (tracker.c.CrossValidationCount > 0)
                             & (tracker.c.SeasonalityPredicted == None) #pylint: disable=singleton-comparison
                             & (tracker.c.AnalysisAttempt < const.MAX_ANALYZE_ATTEMPTS)
                             ).execute().fetchone()[0]

def check_seasonality(row, is_test=False):
    """ Time Series based Seasonality Analysis for selected stock """

    #Check Seasonality
    try:
        ticker = row.ListingTicker
        name = row.ListingName
        #print("Starting Seasonality analysis for Stock: %s with ticker: %s" % (name, ticker))
        LOG.info("Starting Seasonality analysis for Stock: %s with ticker: %s", name, ticker)

        if (row.SeasonalityObserved is None) | is_test:
            #Run CV only when required
            output = crossvalidate(row, is_test)
            #Update status in DB
            tracker.update().\
                where(tracker.c.ForecastID == row.ForecastID).\
                values(SeasonalityObserved=output.seasonality
                       , BestModels=output.models
                       , OutputModelFolder=output.outputModelFolder
                       , OutputDataFolder=output.outputDataFolder
                       , OutputImageFolder=output.outputImageFolder
                       #, AnalysisAttempt=row.AnalysisAttempt + 1
                       , LastModifiedDate=datetime.now()).\
                execute()

        #Run Prediction, else you won't be here
        output = predict(row, is_test)
        #Update status in DB
        tracker.update().\
            where(tracker.c.ForecastID == row.ForecastID).\
            values(SeasonalityPredicted=output.seasonality
                   , ConsistentModels=output.models
                   , OutputModelFolder=output.outputModelFolder
                   , OutputDataFolder=output.outputDataFolder
                   , OutputImageFolder=output.outputImageFolder
                   , LastModifiedDate=datetime.now()
                   , AnalysisAttempt=row.AnalysisAttempt + 1).\
            execute()

    except (TypeError, ValueError, ImportError, MemoryError, IOError) as err:
        LOG.info("Error in Seasonality analysis analysis for Stock: %s with ticker: %s"
                 , name, ticker)
        LOG.error("\nException Line No.: %s  \nMsg: %s"
                  , str(sys.exc_info()[2].tb_lineno), str(err))

        #Update status in DB
        tracker.update().\
            where(tracker.c.ForecastID == row.ForecastID).\
            values(LastModifiedDate=datetime.now()
                   , AnalysisAttempt=row.AnalysisAttempt + 1).\
            execute()
        #raise Exception

    LOG.info("Completed Seasonality analysis for Stock: %s with ticker: %s", name, ticker)

def start_analysis(is_test=False):
    """ Analyzer Code """
    LOG.info("Starting analyzer")

    #Check if row exists for current year
    stock_cnt = tracker.count(tracker.c.ForecastYear == const.CURRENT_YEAR).execute().fetchone()[0]
    LOG.info("No. of stocks listed: %d", stock_cnt)

    pending_cnt = tracker.count((tracker.c.ForecastYear == const.CURRENT_YEAR)
                                & (tracker.c.RawDataFilePath == None) #pylint: disable=singleton-comparison
                                & (tracker.c.DataCollectionAttempt < MAX_FETCH_ATTEMPTS)
                                ).execute().fetchone()[0]
    LOG.info("Raw data collection pending for %d out of total %d stocks."
             , pending_cnt, stock_cnt)

    #Check for row with data available but analysis is pending
    tba_stock_cnt = get_tba_stocks()
    LOG.info("Data transformation pending for %d out of total %d stocks."
             , tba_stock_cnt, stock_cnt)

    retry = True #Change to true
    while retry:
        #Fetch 1 stock for analysis at a time
        tba_stock = get_tba_stocks(1)

        if is_test:
            tba_stock = get_ut_stock()

        if not tba_stock:
            #End while loop after there are no more stocks pending
            retry = False
        else:
            #Time Series based Seasonality Analysis
            transform(tba_stock[0])
            #Execute only 1 cycle in case of unit testing
            if is_test:
                retry = False

    #Check for row with data available but analysis is pending
    pending_stock_cnt = get_tsa_pending_stocks()
    LOG.info("Time series analysis pending for %d out of total %d stocks."
             , pending_stock_cnt, stock_cnt)

    retry = True
    while retry:
        #Fetch 1 stock for analysis at a time
        tsa_stock = get_tsa_pending_stocks(1)

        if is_test:
            tsa_stock = get_ut_stock()

        if not tsa_stock:
            #End while loop after there are no more stocks pending
            retry = False
        else:
            check_seasonality(tsa_stock[0], is_test)
            #Execute only 1 cycle in case of unit testing
            if is_test:
                retry = False

    #Exit routine
    #Close all the figures in plot to avoid memory errors
    #plt.close('all')

    LOG.info("Time Series based Seasonality Analysis completed")
    so_cnt = tracker.count((tracker.c.ForecastYear == const.CURRENT_YEAR)
                           & (tracker.c.SeasonalityObserved != None) #pylint: disable=singleton-comparison
                           & (tracker.c.BestModels == None) #pylint: disable=singleton-comparison
                          ).execute().fetchone()[0]
    LOG.info("No past seasonality is observed for %d out of total %d stocks."
             , so_cnt, stock_cnt)
    sp_cnt = tracker.count((tracker.c.ForecastYear == const.CURRENT_YEAR)
                           & (tracker.c.SeasonalityPredicted != None) #pylint: disable=singleton-comparison
                           & (tracker.c.ConsistentModels == None) #pylint: disable=singleton-comparison
                          ).execute().fetchone()[0]
    LOG.info("No seasonality is predicted for %d out of total %d stocks."
             , sp_cnt, stock_cnt)
    LOG.info("Exiting analyzer")

#Instantiate global variables for this app
#pylint: disable=invalid-name

#Create directory to store output if doesn't already exist
create_dir(const.MODDATA_PATH, ' for transformed data')
create_dir(const.OUTPUT_PATH, ' for analysis output')
create_dir(const.MODEL_PATH, ' for analysis models')
create_dir(const.CSV_PATH, ' for analysis predictions')
create_dir(const.IMG_PATH, ' for analysis plots')

#Check DB connection
db = connect_db(const.SQLALCHEMY_DATABASE_URI, False)
metadata = MetaData(db)

#Create Table object
tracker = get_table('ForecastTracker', metadata)
