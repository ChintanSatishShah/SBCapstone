"""Constants"""
from datetime import datetime, date
import os

from config import Config as conf
from . import APP_ROOT

LOG_FILENAME = 'app.log'
#SQLALCHEMY_DATABASE_URI = r'sqlite:///' + APP_ROOT + r'./db/StockSeasonality.db'

OUTPUT_PATH = os.path.join(APP_ROOT, conf.OUTPUT_DIRECTORY)
CSV_PATH = OUTPUT_PATH + r'/' +  conf.DIR_RESULT
#IMG_PATH = os.path.join(OUTPUT_PATH, conf.DIR_IMAGE)
IMG_PATH = r'static/' + conf.DIR_IMAGE
ROOT = r'./app/'

DATA_SFX = '_data.png'
CV_ALL_SFX = '_cv_all.csv'
CV_BST_SFX = '_cv_best.csv'
CV_PLT_ALL_SFX = '_cv_all.png'
CV_PLT_BST_SFX = '_cv_best.png'
PRED_ALL_SFX = '_pred_all.csv'
PRED_BST_SFX = '_pred_best.csv'
PRED_PLT_ALL_SFX = '_pred_all.png'
PRED_PLT_BST_SFX = '_pred_best.png'

MAX_FETCH_ATTEMPTS = 5
TODAY = date.today()
CURRENT_YEAR = datetime.today().year

# pylint: disable=invalid-name
year_col = 'year'
price_col = 'adj_close'
price_sma = 'sma'
wd_col = 'work_doy'
pctile_col = 'pctile'
rec_cnt = 'record_cnt'
pred_min_dt_col = "Pred. Min. Price Date"
method_col = "Algorithm"
