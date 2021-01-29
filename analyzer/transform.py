""" Transforming and Cleaning Data """

import logging
import pandas as pd

import analyzer.constants as const
from analyzer.constants import year_col, price_col, price_sma, wd_col, pctile_col

# pylint: disable=invalid-name
LOG = logging.getLogger(__name__)

def get_rawdata(file_path):
    """ Get raw data from csv file """
    #Select required columns from csv
    #rawdata = pd.read_csv(file_path, index_col=0, usecols=['Date', 'Adj Close'], parse_dates=True)
    rawdata = None
    rawdata = pd.read_csv(file_path, index_col=0, parse_dates=True)
    rawdata = rawdata.dropna(how='any') ##Drop rows with empty cells
    rawdata.index.names = ['date']
    rawdata.rename(columns={"Adj Close": "adj_close"}, inplace=True)
    #Sort data by date
    rawdata.sort_index(ascending=True, inplace=True)
    return rawdata.copy()

def transform_data(rawdata_df):
    """ Transofrm raw data to required format """
    rawdf = rawdata_df.copy()

    #Remove listing year rows
    rawdf = rawdf[rawdf.index.year > rawdf.index.year.min()]

    #Simple moving average of adjusted close price with 2 prior and 2 next values
    rawdf['sma'] = rawdf.adj_close.\
                             rolling(window=const.sma_rolling_window, min_periods=1).\
                             mean().shift(-2)
    rawdf[year_col] = rawdf.index.year
    #Drop current year's records from analysis
    rawdf = rawdf[rawdf.year < const.CURRENT_YEAR]
    rawdf["month"] = rawdf.index.month
    rawdf['week'] = pd.Int64Index(rawdf.index.isocalendar().week)
    rawdf["day_of_week"] = rawdf.index.dayofweek
    rawdf["day_of_month"] = rawdf.index.day
    #Store day of year
    rawdf['day_of_year'] = rawdf.index.dayofyear
    #Add row count by year to get working day of year, column used to align prediction output
    rawdf[wd_col] = rawdf.groupby([year_col]).cumcount() + 1
    #Add percentile using rank, this is column used for determining the accuracy of prediction
    rawdf[pctile_col] = 100 * rawdf.groupby(year_col)[price_sma].\
                              rank("min", pct=True, ascending=True)
    #Handle ISO week format for last days of year marked as Week 1
    rawdf.loc[(rawdf['week'] == 1) & (rawdf['day_of_year'] > 350), 'week'] = 53
    return rawdf.copy()

def agg_data(mod_df, wd_dict):
    """ Derive important yearly metrics """
    #Aggregate data on yearly basis - Min. day of year will be our output
    yagg_df = None
    yagg_df = pd.concat([mod_df.groupby(year_col)[price_col].count().rename('record_cnt')
                         , mod_df.groupby(year_col)[price_col].mean().rename('mean_' + price_col)
                         , mod_df.groupby(year_col)[price_col].min().rename('min_' + price_col)
                         , mod_df.groupby(year_col)[price_sma].min().rename('min_' + price_sma)
                         , mod_df.groupby(year_col)[price_col].idxmin().\
                              rename('date_min_' + price_col)
                         , mod_df.groupby(year_col)[price_sma].idxmin().\
                              rename('date_min_' + price_sma)
                         , mod_df.groupby(year_col)[price_col].idxmin().dt.dayofyear.\
                              rename('doy_min_' + price_col)
                         , mod_df.groupby(year_col)[price_sma].idxmin().dt.dayofyear.\
                              rename('doy_min_' + price_sma)
                         #, mod_df.groupby(year_col)[price_col].max()
                         #, mod_df.groupby(year_col)[price_col].median()
                         ], axis=1)
    ## Remove rows for years when the adj_close has -ve values
    #TO DO - This code needs to be revised
    #get the list of years where min. price < 0
    #remove data before the max. value of year in that list.
    yagg_df = yagg_df[yagg_df.min_adj_close > 0]

    ##Get working day of min. price
    yagg_df['wd_min_'+price_col] = yagg_df.apply(lambda row: wd_dict[row.date_min_adj_close]
                                                 , axis=1)
    yagg_df['wd_min_'+price_sma] = yagg_df.apply(lambda row: wd_dict[row.date_min_sma], axis=1)

    ## Check diff. between min. price working day based on actual value and moving avg.
    yagg_df['target_doy_diff'] = abs(yagg_df.doy_min_adj_close - yagg_df.doy_min_sma)
    yagg_df['target_wd_diff'] = abs(yagg_df.wd_min_adj_close - yagg_df.wd_min_sma)
    #return yagg_df.copy()
    return yagg_df
