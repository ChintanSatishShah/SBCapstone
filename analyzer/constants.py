"""Constants"""
from datetime import datetime, date
import os.path
import pandas as pd
from analyzer.conf import AnalyzerConfig as conf

#ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../')
#APP_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../')
APP_ROOT = r''
LOG_FILENAME = 'analyzer.log'
SQLALCHEMY_DATABASE_URI = r'sqlite:///' + APP_ROOT + r'./db/StockSeasonality.db'

RAWDATA_PATH = os.path.join(APP_ROOT, conf.DIR_RAWDATA)
MODDATA_PATH = os.path.join(APP_ROOT, conf.DIR_MOD)
OUTPUT_PATH = os.path.join(APP_ROOT, conf.OUTPUT_DIRECTORY)
MODEL_PATH = os.path.join(OUTPUT_PATH, conf.DIR_MODEL)
CSV_PATH = os.path.join(OUTPUT_PATH, conf.DIR_RESULT)
IMG_PATH = os.path.join(APP_ROOT, conf.WEBAPP_DIRECTORY, conf.DIR_IMAGE)
FILE_YR_AGG = '_yagg.csv'

IMG_TYPE = '.png'
IMG_DATA = '_data' + IMG_TYPE
IMG_PREV = '_prev' + IMG_TYPE
IMG_PRED = '_pred' + IMG_TYPE

#NSE = 'NSE'
#LISTING_TYPE = 'Equity'
#SYSTEM = 'System'
MAX_FETCH_ATTEMPTS = 5
MAX_ANALYZE_ATTEMPTS = 3
TODAY = date.today()
CURRENT_YEAR = datetime.today().year

# pylint: disable=invalid-name
#Initialize the common variables
sma_rolling_window = 5 #Rolling window for simple moving average of stock price
min_trn_yrs = 5
min_cross_val = 3
max_cross_val = 5
min_years_data_reqd = min_trn_yrs + min_cross_val
#print("Min.", min_years_data_reqd, "years of historical stock data required for analysis")

year_col = 'year'
price_col = 'adj_close'
price_sma = 'sma'
wd_col = 'work_doy'
pctile_col = 'pctile'
rec_cnt = 'record_cnt'
pred_wd_col = 'pred_wd_min'
pred_dt_col = "Pred. Min. Price Date"
method_col = "Algorithm"

#1.25 is the average ratio of error to deviation for marking outliers
outlier_ratio = 1.25
pctile_chk = 20
min_yrs_chk = 2
pred_range = 22 #1 month = @22 working days
date_features = ["year", "month", "week", "day_of_week", "day_of_month", "day_of_year"]
trading_holidays = ['2020-02-21', '2020-03-10', '2020-04-02', '2020-04-06', '2020-04-10'
                    , '2020-04-14', '2020-05-01', '2020-05-25', '2020-10-02', '2020-11-16'
                    , '2020-11-30', '2020-12-25']
pred_dates = pd.date_range(datetime(CURRENT_YEAR, 1, 1), datetime(CURRENT_YEAR, 12, 31),
                           freq=pd.tseries.offsets.BDay()).to_frame(index=False, name='ds')
pred_dates = pred_dates[~pred_dates.ds.isin(trading_holidays)]
#print('No. of working days in', CURRENT_YEAR, ':', len(pred_dates))
pred_dates.reset_index(drop=True, inplace=True)
pred_dates.index += 1
pred_dates.index.names = ['work_doy']

#Algorithms
ALGO_FILE = os.path.join(APP_ROOT, './db/algorithms.csv')
algos_df = pd.read_csv(ALGO_FILE, index_col=0)
#print(algos_df)
