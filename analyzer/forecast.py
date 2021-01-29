""" prediction for Analyzer """

#Import the necessary packages
import logging
import os
import warnings
import numpy as np
import pandas as pd

#Statsmodels
#from statsmodels.tsa.stattools import adfuller
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES

#PMD Arima
import pmdarima as pm
from pmdarima.metrics import smape
#print(f"Using pmdarima {pm.__version__}")

#FB Prophet
from fbprophet import Prophet

#Prediction Accuracy Metrics
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import mean_squared_error

import analyzer.constants as const
from analyzer.constants import min_trn_yrs, max_cross_val, outlier_ratio, pctile_chk, min_yrs_chk
from analyzer.constants import year_col, price_col, wd_col, rec_cnt, pred_range, pred_wd_col
from analyzer.constants import date_features, algos_df, pred_dates, pred_dt_col, method_col
from analyzer.plot import plot_cv

# pylint: disable=invalid-name
pd.options.display.precision = 6
logging.getLogger('fbprophet').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')
LOG = logging.getLogger(__name__)

class processing_row():
    """ Represents a Stock processing row """
    def __init__(self, ticker, name, attempt, unit_test, max_pred_price=None, cross_val=None
                 , split_year=None, frequency=None, diff=None, sdiff=None, is_pred=False):
        self.ticker = ticker
        self.name = name
        self.attempt = attempt
        self.unit_test = unit_test
        self.max_pred_price = max_pred_price
        self.cross_val = cross_val
        self.split_year = split_year
        self.frequency = frequency
        self.diff = diff
        self.sdiff = sdiff
        self.is_prediction = is_pred

    #def __repr__(self):
        #return str(self.__dict__)

    #def get_object_csv(self):
        #""" Get the object as csv """
        #return ', '.join(map(str, self.__dict__.values()))

class result_row():
    """ Represents a result of Stock analysis """
    def __init__(self, ticker, name, seasonality, models
                 , outputModelFolder, outputDataFolder, outputImageFolder):
        self.ticker = ticker
        self.name = name
        self.seasonality = seasonality
        self.models = models
        self.outputModelFolder = outputModelFolder
        self.outputDataFolder = outputDataFolder
        self.outputImageFolder = outputImageFolder

    #def __repr__(self):
        #return str(self.__dict__)

    #def get_object_csv(self):
        #""" Get the object as csv """
        #return ', '.join(map(str, self.__dict__.values()))

def get_value(irow, series):
    """ Get working day for given dates """
    #print(irow)
    ref = pd.Series(index=irow.index)
    cnt = len(irow)
    for i in range(cnt):
        ref.loc[irow.index[i]] = (series[irow[i]])
    #print(ref)
    return ref

def get_pctile_fron_wd(prow, df):
    """ Get percentile from working day and year """
    ref = prow.copy()#pd.Series(index = prow.index)
    #print(ref)
    year = prow[0]
    #print(year)
    prow = prow[1:]
    #print(prow)
    cnt = len(prow)
    for i in range(cnt):
        #print(prow[i])
        ref.loc[prow.index[i]] = df[(df.year == year) & (df[wd_col] == round(prow[i], 0))].pctile[0]
    #print(ref)
    return ref

def get_date_fron_wd(prow, df):
    """ Get percentile from working day and year """
    ref = prow.copy()#pd.Series(index = prow.index)
    #print(ref)
    year = prow[0]
    #print(year)
    prow = prow[1:]
    #print(prow)
    cnt = len(prow)
    for i in range(cnt):
        #print(prow[i])
        ref.loc[prow.index[i]] = df[(df.year == year) & (df[wd_col] == round(prow[i], 0))].\
            index[0].strftime("%d %b") \
            + "~" \
            + str(round((df[(df.year == year) & (df[wd_col] == round(prow[i], 0))].pctile[0]), 2))
    #print(ref)
    return ref

def get_desc_from_code(ref):
    """ Get Algorithm description from code """
    #print(ref)
    desc = algos_df.loc[(algos_df.CodeName == ref)].Description.values[0]
    #print(desc)
    return desc

#def pred_output(stock, df):
    #""" Remove outliers from min. price date predictions using mean (z-stats) """
    ##Older method#
    #LOG.info("Checking for consistency in predictions for: %s", stock.ticker)
    #LOG.info(str(df))
    #retry = 0

    #Check for closely matching dates about 2 weeks period
    #while ((df.max()[0] - df.min()[0]) > 10) & (retry < 10):
        ##Increase outlier ratio on each loop i.e. 1.25, 1.75, 2.25...
        ##prev_df = df
        #otlr_ratio = outlier_ratio + (0.5 * retry)
        #print(str(df.sub(df.mean()).div(df.std()).abs()))
        #df = df * df.sub(df.mean()).div(df.std()).abs().lt(otlr_ratio)
        #print(df)
        ##Remove outliers indicated by 0
        #df = df[df.pred_wd_min > 0]
        #print(df)
        #retry = retry + 1
        #print(retry)
        #LOG.info("Cycle %d: \n %s", retry, str(df))
        #if prev_df.equals(df):
            #break
        #else:
        #continue

    #return df.sort_values(by=pred_wd_col)

def align_pred_output(stock, df):
    """ Remove outliers from min. price date predictions using median """
    LOG.info("Checking for consistency in predictions for: %s", stock.ticker)
    LOG.info(str(df))
    # Absolute value means effectively double range
    lst_days_diff = [45, 30, 15, 10, 5]
    #Check for closely matching dates about 2 weeks period
    for days_diff in lst_days_diff:
        #Increase outlier ratio on each loop i.e. 1.25, 1.75, 2.25...
        prev_df = df.copy()
        #print(str(df.sub(df.median()).abs()))
        df = df * df.sub(df.median()).abs().lt(days_diff)
        #print(df)
        #Remove outliers indicated by 0
        df = df[df.pred_wd_min > 0]
        #print(df)
        LOG.info("Check for absoulte range %d: \n %s", days_diff, str(df))
        if (df.max()[0] - df.min()[0]) < pred_range: #(2 * lst_days_diff[-1]):
            #Break the loop if required min. date range is achieved
            LOG.info("Closing Analysis since min. date range is achieved")
            break
        elif len(df) <= 5:
            #Break the loop if count of algos reduces to under 5
            #Atleast 3 algos should give consistent results
            if len(df) < 3:
                df = prev_df.copy()
            LOG.info("Closing Analysis since min. algorithm count is achieved")
            break
        else:
            continue

    return df.sort_values(by=pred_wd_col)

def get_aa_diff(y_df, stock):
    """ Directly estimate the number of differences required """
    kpss_diffs = pm.arima.ndiffs(y_df, alpha=0.05, test='kpss', max_d=6)
    adf_diffs = pm.arima.ndiffs(y_df, alpha=0.05, test='adf', max_d=6)
    pp_diffs = pm.arima.ndiffs(y_df, alpha=0.05, test='pp', max_d=6)
    n_diffs = max(adf_diffs, kpss_diffs, pp_diffs)

    #print(f"Estimated differencing term, d: {n_diffs}")
    LOG.info("Estimated differencing term, d: %d", n_diffs)

    ##directly estimate the number of seasonal differences
    ocsb_sdiffs = pm.arima.nsdiffs(y_df, m=stock.frequency, test='ocsb', max_D=6)
    ch_sdiffs = pm.arima.nsdiffs(y_df, m=stock.frequency, test='ch', max_D=6)
    n_sdiffs = max(ocsb_sdiffs, ch_sdiffs)

    #print(f"Estimated seasonal differencing term, D: {n_sdiffs}")
    LOG.info("Estimated seasonal differencing term, D: %d", n_sdiffs)
    stock.simple_diff = n_diffs
    stock.seasonal_diff = n_sdiffs
    return stock

def set_future_exog_data(data_df, pred_df, freq):
    """ Set Exogenous data for Prediction """
    exog_data = pred_dates.copy()
    #exog_data.reset_index(inplace = True) #Uncomment if work day is required
    exog_data.set_index('ds', inplace=True)
    exog_data['year'] = exog_data.index.year
    exog_data["month"] = exog_data.index.month
    exog_data['week'] = pd.Int64Index(exog_data.index.isocalendar().week)
    exog_data["day_of_week"] = exog_data.index.dayofweek
    exog_data["day_of_month"] = exog_data.index.day
    exog_data['day_of_year'] = exog_data.index.dayofyear
    #Handle ISO week format for last days of year marked as Week 1
    exog_data.loc[(exog_data['week'] == 1) & (exog_data['day_of_year'] > 350), 'week'] = 53
    #print(exog_data)
    exog_data[price_col] = pred_df.values
    #print(exog_data[price_col].values)

    full_df = pd.concat([data_df[[price_col] + date_features]
                         , exog_data[[price_col] + date_features]])

    #FourierFeaturizer
    #k = no. of Sin/Cos terms - columns = k*2
    trans = pm.preprocessing.FourierFeaturizer(m=freq, k=4)
    y_prime, ff_data = trans.fit_transform(full_df[price_col], exogenous=full_df[date_features])
    #print("y_prime: " + str(type(y_prime)))
    LOG.info("y_prime: %s", str(type(y_prime)))

    return ff_data

def cv_sd(stock, data_df, yagg_df, wd_series, file_path):
    """ Seasonal Decompose based cross-validation """
    method = "Seasonal Decompose"
    LOG.info("Starting %s based cross-validation for %s", method, stock.ticker)
    algos = algos_df[(algos_df.Method.str.startswith(method)) & (algos_df.IsActive == 1)]

    if stock.unit_test:
        algos = algos[algos.UnitTest == 1]

    if not algos.empty: #Control Execution for debugging
        start_from = stock.split_year + 1 - min_trn_yrs
        sd_data = data_df[data_df.year >= start_from - 1][[year_col]]

        #Running seasonal_decompose for cross validation years
        for i in range(stock.cross_val + min_trn_yrs - 1):
            pred_year = start_from + i
            y_train = data_df[data_df.year < pred_year][price_col]
            train_years = len(y_train.index.year.unique())
            freq = int(round((len(y_train) / train_years), 0))
            yr_rec_cnt = len(data_df[data_df.year == y_train.index.year.max()])

            #Check for min. SD requirement - must have atleast 2 complete cycles of data
            if len(y_train) >= (2 * freq):
                for algo in algos.itertuples():
                    if algo.Library == 'statsmodels':
                        #seasonal_decompose using statsmodels package
                        sd = seasonal_decompose(y_train
                                                , model=algo.Seasonality.lower(), period=freq)
                        sd_data.loc[y_train.index[-yr_rec_cnt:], algo.CodeName] = pd.Series\
                            (sd.seasonal[-yr_rec_cnt:], index=y_train.index[-yr_rec_cnt:])
                    else:
                        #seasonal_decompose using pmdarima package
                        pmd_sd = pm.arima.decompose(y_train.values
                                                    , type_=algo.Seasonality.lower()
                                                    , m=freq, filter_=None)
                        sd_data.loc[y_train.index[-yr_rec_cnt:], algo.CodeName] = pd.Series\
                            (pmd_sd.seasonal[-yr_rec_cnt:], index=y_train.index[-yr_rec_cnt:])

        sd_data.dropna(inplace=True)
        #print(sd_data.head())
        #print(sd_data.tail())
        sd_data.drop([year_col], axis=1).to_csv(file_path, header=True, index=True)

        #Get year-wise min. price
        sd_agg = sd_data.groupby(year_col).idxmin().dropna()
        sd_agg = sd_agg.apply(get_value, axis=1, args=(wd_series,))
        #print(sd_agg)

        #Get avg. seasonality w/o removing outliers to show the diff. betn. with vs w/o outliers
        #sd_rm = sd_agg.rolling(min_trn_yrs).mean().dropna()
        #print(sd_rm)

        #Get avg. seasonality for last 5 years while ignoring the outlier years
        #sd_refined = df * df.sub(df.rolling(5).mean()).div(df.rolling(5).std()).abs().lt(1.25)
        sd_refined = pd.DataFrame(columns=sd_agg.columns)
        for i in range(stock.cross_val):
            pred_year = stock.split_year + i
            #print(pred_year)
            df = sd_agg.loc[pred_year-5:pred_year-1]
            #print(df)
            # TO TRY - ignore std < 5, outlier_diff = 5
            #print(df.mean(), df.std(), df.sub(df.mean()).div(df.std()).abs())
            ref_df = df * df.sub(df.mean()).div(df.std(), ).abs().lt(outlier_ratio)
            #Replace 0 with NaN to ensure that outliers are ignored while calculating mean
            ref_df.replace(0, np.nan, inplace=True)
            #print(ref_df)
            #print(ref_df.mean())
            sd_refined = sd_refined.append(ref_df.mean(), ignore_index=True)

        #print(sd_refined)
        #Increment the year by 1 to indicate that this is the prediction for next year
        sd_refined.index += stock.split_year
        #Add the actual output column - wd_min_sma
        sd_refined = yagg_df[['wd_min_sma']].join(sd_refined, how='inner')
        #Rename column to indicate the correct interpretation
        sd_refined.index.name = 'pred_year'
        #print(sd_refined)

        #sd_pctile = sd_refined.reset_index()
        #sd_pctile = sd_pctile.apply(get_pctile_fron_wd, axis=1, args=(data_df,))
        #sd_pctile.set_index('pred_year', inplace=True)
        #print(sd_pctile)

        LOG.info("Completed Seasonal Decompose based cross-validation for %s", stock.ticker)
        return sd_refined

def ts_fcast(stock, data_df, yagg_df, ff_df, file_path):
    """ Time-Series based prediction/cross-validation """
    if stock.is_prediction:
        process = "based prediction"
        ts_data = pred_dates.copy()
        split_yr = const.CURRENT_YEAR
    else:
        split_yr = stock.split_year
        process = "based cross-validation"
        ts_data = data_df[data_df.year >= split_yr][[year_col, price_col]]
        ts_metrics = pd.DataFrame(index=[*range(split_yr, const.CURRENT_YEAR)])

    LOG.info("Starting Time-Series %s for %s", process, stock.ticker)

    algos = algos_df[(algos_df.Method.str.startswith('Time Series')) & (algos_df.IsActive == 1)]
    if stock.unit_test:
        algos = algos[algos.UnitTest == 1]

    if not algos.empty: #Control Execution for debugging

        #Setting up dataset as per FB prophet requirements
        pht_data = pd.DataFrame()
        pht_data['ds'] = data_df.index
        pht_data['y'] = data_df[price_col].values
        #Required for growth = "logistic"
        pht_data['cap'] = 1.5 * pht_data.y.max() #Capping the price growth at 50% of max. value

        #Running Time Series prediction for cross validation years
        for i in range(stock.cross_val):
            pred_year = split_yr + i
            train_data = data_df[data_df.year < pred_year][[price_col] + date_features]
            y_train = train_data[price_col] #data_df[data_df.year < pred_year][price_col]
            train_years = len(y_train.index.year.unique())
            freq = int(round((len(y_train) / train_years), 0))

            #Setting up dataset as per FB prophet requirements
            py_train = pht_data[pht_data.ds.dt.year < pred_year]

            if not stock.is_prediction:
                test_data = data_df[data_df.year == pred_year][[price_col] + date_features]
                y_test = test_data[price_col] #data[data.year == pred_year][price_col]
                py_test = pht_data[pht_data.ds.dt.year == pred_year]
            else:
                y_test = ts_data.copy()

            for algo in algos.itertuples():
                code = algo.CodeName

                if algo.Library == 'statsmodels':
                    #Holt-Winters Exponential Smoothing - HWES based Cross Validation
                    method = "Holt-Winters Exponential Smoothing"
                    LOG.info("Starting %s %s for stock:%s, year:%d, code:%s"
                             , method, process, stock.ticker, pred_year, code)
                    mode = algo.Seasonality.lower()
                    if algo.Trend == 'None':
                        trend = None
                    else:
                        trend = algo.Trend.lower()

                    hwes_model = HWES(y_train, seasonal_periods=freq
                                      , trend=trend, seasonal=mode).fit()
                    #print(hwes_model.summary())
                    if stock.is_prediction:
                        fc_steps = len(pred_dates)
                    else:
                        fc_steps = yagg_df.loc[pred_year, rec_cnt]

                    hwes_fc = hwes_model.forecast(steps=fc_steps)

                    ts_data.loc[y_test.index, code] = pd.Series(hwes_fc.values, index=y_test.index)
                    if not stock.is_prediction:
                        #Set CV metrics
                        ts_metrics.loc[pred_year, f'smape_{code}'] = smape(y_test, hwes_fc)

                    LOG.info("Completed %s %s for stock:%s, year:%d, code:%s"
                             , method, process, stock.ticker, pred_year, code)
                elif algo.Library == 'pmdarima':
                    #Auto-ARIMA based Cross Validation
                    method = "Auto-ARIMA"
                    LOG.info("Starting %s %s for stock:%s, year:%d, code:%s"
                             , method, process, stock.ticker, pred_year, code)

                    if stock.is_prediction:
                        #Any of the prev. algorithm outputs as exogenous input for ARIMAX
                        #Select algo such that Unit test is not broken
                        pred_algo = 'ts_hwes_non_add'
                        #FourierFeaturizer
                        ff_df = set_future_exog_data(data_df, ts_data[pred_algo], freq)

                    train_exog = None
                    test_exog = None
                    if "date" in code:
                        if stock.attempt == 0:
                            #Drop other exogenous features, if added in future
                            train_exog = ff_df[ff_df.year < pred_year]
                            test_exog = ff_df[ff_df.year == pred_year]
                        else:
                            # Code to avoid SARIMAX error on retry attempts - Drop year column
                            # A constant trend was included in the model specification,
                            # but the `exog` data already contains a column of constants.
                            train_exog = ff_df[ff_df.year < pred_year].drop([year_col], axis=1)
                            test_exog = ff_df[ff_df.year == pred_year].drop([year_col], axis=1)
                    else:
                        #Drop date based exogenous features
                        train_exog = ff_df[ff_df.year < pred_year].drop(date_features, axis=1)
                        test_exog = ff_df[ff_df.year == pred_year].drop(date_features, axis=1)

                    aa_model = pm.auto_arima(y_train, exogenous=train_exog
                                             , d=stock.diff, max_p=6, seasonal=False
                                             , stepwise=True, suppress_warnings=True, trace=False)
                    #print('Model order:', aa_model.order)
                    LOG.info("Auto-ARIMA Model order: %s", aa_model.order)

                    #print(aa_model.summary())
                    aa_fcast = aa_model.predict(n_periods=len(y_test), exogenous=test_exog
                                                , return_conf_int=False)

                    ts_data.loc[y_test.index, code] = pd.Series(aa_fcast, index=y_test.index)

                    if not stock.is_prediction:
                        #Set CV metrics
                        ts_metrics.loc[pred_year, f'smape_{code}'] = smape(y_test, aa_fcast)

                    LOG.info("Completed %s %s for stock:%s, year:%d, code:%s"
                             , method, process, stock.ticker, pred_year, code)

                else:
                    #Facebook - Prophet based Cross Validation
                    method = "Prophet(Facebook)"
                    LOG.info("Starting %s %s for stock:%s, year:%d, code:%s"
                             , method, process, stock.ticker, pred_year, code)
                    mode = algo.Seasonality.lower()
                    trend = algo.Trend.lower()

                    if stock.is_prediction:
                        py_test = pred_dates.copy()
                        #Capping the price growth at 50% of max. value
                        py_test['cap'] = 1.5 * pht_data.y.max()

                    fb_model = Prophet(yearly_seasonality=True, weekly_seasonality=False
                                       , growth=trend, seasonality_mode=mode)
                    fb_model.fit(py_train)
                    fb_fcast = fb_model.predict(py_test)

                    ts_data.loc[y_test.index, code] = pd.Series(fb_fcast.yhat.values
                                                                , index=y_test.index)
                    ts_data.loc[y_test.index, f'sd_{code}'] = pd.Series(fb_fcast.yearly.values
                                                                        , index=y_test.index)
                    if not stock.is_prediction:
                        #Set CV metrics
                        ts_metrics.loc[pred_year, f'smape_{code}'] = smape(y_test, fb_fcast.yhat)

                    LOG.info("Completed %s %s for stock:%s, year:%d, code:%s"
                             , method, process, stock.ticker, pred_year, code)

                #Use break to stop execution after 1 loop for debugging
                #break
            #break

        #Save fcast data
        ts_data.to_csv(file_path, header=True, index=True)
        LOG.info("Completed Time-Series %s for %s", process, stock.ticker)

        #print(ts_data)
        if stock.is_prediction:
            return None
        else:
            #print(ts_metrics)
            #LOG.info("Metrics of Time-Series %s for %s: \n %s"
                     #, process, stock.ticker, str(ts_metrics))
            return ts_metrics

def pred_sd(stock, data_df, wd_series, file_path):
    """ Seasonal Decompose based Time-Series prediction """
    method = "Seasonal Decompose"
    LOG.info("Starting %s based prediction for %s", method, stock.ticker)
    algos = algos_df[(algos_df.Method.str.startswith(method)) & (algos_df.IsActive == 1)]

    if stock.unit_test:
        algos = algos[algos.UnitTest == 1]

    if not algos.empty: #Control Execution for debugging
        start_from = const.CURRENT_YEAR + 1 - min_trn_yrs
        sd_data = data_df[data_df.year >= start_from - 1][[year_col]]
        #sd_data = data_df[data_df.year >= stock.split_year][[year_col]] #Different

        #Running seasonal_decompose for Min. training years
        for i in range(min_trn_yrs): #Different
            pred_year = start_from + i
            y_train = data_df[data_df.year < pred_year][price_col]
            #pred_year = stock.split_year + i #Different
            #y_train = data_df[data_df.year <= pred_year][price_col] #Different
            train_years = len(y_train.index.year.unique())
            freq = int(round((len(y_train) / train_years), 0))
            yr_rec_cnt = len(data_df[data_df.year == y_train.index.year.max()])

            for algo in algos.itertuples():
                if algo.Library == 'statsmodels':
                    #seasonal_decompose using statsmodels package
                    sd = seasonal_decompose(y_train, model=algo.Seasonality.lower(), period=freq)
                    sd_data.loc[y_train.index[-yr_rec_cnt:], algo.CodeName] = pd.Series\
                        (sd.seasonal[-yr_rec_cnt:], index=y_train.index[-yr_rec_cnt:])
                else:
                    #seasonal_decompose using pmdarima package
                    pmd_sd = pm.arima.decompose(y_train.values
                                                , type_=algo.Seasonality.lower()
                                                , m=freq, filter_=None)
                    sd_data.loc[y_train.index[-yr_rec_cnt:], algo.CodeName] = pd.Series\
                        (pmd_sd.seasonal[-yr_rec_cnt:], index=y_train.index[-yr_rec_cnt:])

        sd_data.dropna(inplace=True)
        #print(sd_data.head())
        #print(sd_data.tail())
        sd_data.drop([year_col], axis=1).to_csv(file_path, header=True, index=True)

        #Get year-wise min. price
        sd_agg = sd_data.groupby(year_col).idxmin().dropna()
        sd_agg = sd_agg.apply(get_value, axis=1, args=(wd_series,))
        #print(sd_agg)

        sd_refined = pd.DataFrame(index=[const.CURRENT_YEAR], columns=sd_agg.columns)

        df = sd_agg.copy()
        #Doesn't matter even if some values are dropped in case of lower stdev below 5
        ref_df = df * df.sub(df.mean()).div(df.std()).abs().lt(outlier_ratio)

        #Replace 0 with NaN to ensure that outliers are ignored while calculating mean
        ref_df.replace(0, np.nan, inplace=True)
        #print(ref_df)
        #print(ref_df.mean())
        sd_refined.loc[const.CURRENT_YEAR] = round(ref_df.mean(), 0)

        #Rename column to indicate the correct interpretation
        sd_refined.index.name = 'pred_year'
        #print(sd_refined)

        LOG.info("Completed %s based prediction for %s", method, stock.ticker)
        return sd_refined

def crossvalidate(row, is_test=False):
    """ Seasonality validation for previous years """

    #Normal flow
    current = processing_row(row.ListingTicker, row.ListingName, row.AnalysisAttempt, is_test)

    ticker = current.ticker
    LOG.info("Starting Seasonality cross-validation for %s", current.ticker)

    #Load data_df from shared drive
    data_df = None
    yagg_df = None
    data_df = pd.read_csv(const.MODDATA_PATH + ticker + '.csv', index_col=0, parse_dates=True)
    yagg_df = pd.read_csv(const.MODDATA_PATH + ticker + const.FILE_YR_AGG, index_col=0)
    wd_series = data_df[wd_col]

    years_of_data = len(yagg_df)
    #current.max_pred_price = data_df[price_col].max() * 2
    current.max_pred_price = data_df[data_df.year >= const.CURRENT_YEAR - min_trn_yrs]\
        [price_col].max() * 2
    current.cross_val = min(years_of_data - min_trn_yrs, max_cross_val, row.CrossValidationCount)
    #print('No. of cross validations:', current.cross_val)
    LOG.info("No. of cross validations: %d", current.cross_val)

    if current.unit_test:
        current.cross_val = 1
        #print('Reduced no. of cross validations:', current.cross_val)
        LOG.info("Reduced no. of cross validations: %d", current.cross_val)

    #print('Train data period:', len(yagg_df) - current.cross_val, 'years')
    LOG.info("Train data period: %d years", len(yagg_df) - current.cross_val)
    current.frequency = int(round(len(data_df) / years_of_data, 0))
    #print('Average no. of records per year:', current.frequency)
    LOG.info("Average no. of records per year: %d", current.frequency)
    current.split_year = const.CURRENT_YEAR - current.cross_val
    #print('Cross validation begins from:', current.split_year)
    LOG.info("Cross validation begins from: %d", current.split_year)

    cv_model_path = const.MODEL_PATH + ticker # + '_cv.pkl'
    cv_sd_path = const.CSV_PATH + ticker + '_cv_sd.csv'
    cv_fc_path = const.CSV_PATH + ticker + '_cv_fc.csv'
    cv_all_path = const.CSV_PATH + ticker + '_cv_all.csv'
    cv_best_path = const.CSV_PATH + ticker + '_cv_best.csv'
    cv_plot_all_path = const.IMG_PATH + ticker + '_cv_all.png'
    cv_plot_bst_path = const.IMG_PATH + ticker + '_cv_best.png'

    #Seasonal Decompose based Cross Validation
    sd_refined = cv_sd(current, data_df, yagg_df, wd_series, cv_sd_path)
    #print(sd_refined)

    if (not os.path.exists(cv_fc_path)) | is_test:

        #Estimate the number of differences required for Auto-ARIMA
        current = get_aa_diff(data_df[price_col].copy(), current)

        #FourierFeaturizer
        #k = no. of Sin/Cos terms - columns = k*2
        trans = pm.preprocessing.FourierFeaturizer(m=current.frequency, k=4)
        y_prime, ff_data = trans.fit_transform(data_df[price_col]
                                               , exogenous=data_df[date_features])
        #print("y_prime: " + str(type(y_prime)))
        LOG.info("y_prime: %s", str(type(y_prime)))

        cv_accuracy = ts_fcast(current, data_df, yagg_df, ff_data, cv_fc_path)
        cv_accuracy.loc['mean'] = cv_accuracy.mean()
        #print(cv_accuracy)
        LOG.info("Prev. years Prediction Metrics: \n %s", str(cv_accuracy))

    #Read output data from csv
    cv_sd_data = pd.read_csv(cv_sd_path, index_col=0, parse_dates=True)
    cv_fcast = pd.read_csv(cv_fc_path, index_col=0, parse_dates=True)

    if not os.path.exists(cv_plot_all_path):
        plot_cv(stock=current, sd_df=cv_sd_data, output_df=cv_fcast
                , file_path=cv_plot_all_path, lalgos=None)

    #Analyze output of different methods
    LOG.info("Starting comparison of cross-validation outputs of different methods for %s"
             , ticker)

    cv_refined = cv_fcast.groupby([cv_fcast.index.year]).idxmin()
    #Remove algorithms where results are Null values
    cv_refined = cv_refined.dropna(axis=1)
    cv_refined = cv_refined.apply(get_value, axis=1, args=(wd_series,))
    cv_min_wd = sd_refined.join(cv_refined)
    cv_min_wd = cv_min_wd.drop([year_col, price_col], axis=1)
    #cv_min_wd.to_csv(cv_all_path, header=True, index=True)
    #print(cv_min_wd)

    cv_pctile = cv_min_wd.reset_index()
    cv_pctile = cv_pctile.apply(get_pctile_fron_wd, axis=1, args=(data_df,))
    cv_pctile.set_index('pred_year', inplace=True)
    #print(cv_pctile)

    #Save Min. price date data for all algos
    cv_min_dt = cv_min_wd.reset_index()
    cv_min_dt = cv_min_dt.apply(get_date_fron_wd, axis=1, args=(data_df,))
    cv_min_dt.set_index('pred_year', inplace=True)
    cv_all = cv_min_dt.copy()
    cv_all.index.names = ['Year']
    all_cols = ["Actual Min. Date"]
    for code in cv_all.columns[1:]:
        all_cols = all_cols + [algos_df.loc[(algos_df.CodeName == code)].Description.values[0]]
    cv_all.columns = all_cols
    cv_all.index = cv_all.index.astype(int)
    #print(cv_all.transpose())
    cv_all.transpose().to_csv(cv_all_path, header=True, index=True)

    #Check for accuracy
    #Predicted min. work day price < percentile threshold of the yearly price variations
    cv_pct = cv_pctile.drop(['wd_min_sma'], axis=1)#Need this column below
    all_algos = cv_pct[cv_pct <= pctile_chk].count().sort_values(ascending=False)
    #print(all_algos)
    top_algos = all_algos[all_algos >= min(min_yrs_chk, current.cross_val)]
    best_algos = ''
    ssn_obvd = ("Seasonality trend observed in stock price of %s for prev. %d years"
                % (current.ticker, current.cross_val))

    if not top_algos.empty:
        #print(list(top_algos.index.values))
        lst_algos = list(top_algos.index.values)
        best_columns = ["Actual Min. Date"]
        best_algos = "Following algorithms gave consistent seasonality predictions"
        best_algos = best_algos + f" over the last {current.cross_val} years:"
        best_df = algos_df[algos_df['CodeName'].isin(lst_algos)]
        #print(best_df)
        for algo in best_df.itertuples():
            best_algos = best_algos + '\n' + algo.Description
            best_columns = best_columns + [algo.Description]
        #print(best_algos)
        LOG.info("Best Algorithms from Prev. years: %s", str(best_algos))
        #Save plot for best algos
        plot_cv(stock=current, sd_df=cv_sd_data, output_df=cv_fcast
                , file_path=cv_plot_bst_path, lalgos=lst_algos)
        #Save data for best algo
        #cv_min_wd[['wd_min_sma'] + lst_algos].to_csv(cv_best_path, header=True, index=True)
        #Save Min. price date data for best algos
        cv_best = cv_min_dt * cv_pctile.copy().apply(lambda x: x < pctile_chk)
        #cv_best.replace(0, np.nan, inplace=True)
        cv_best = cv_best[['wd_min_sma'] + lst_algos]
        cv_best.columns = best_columns
        cv_best.index = cv_best.index.astype(int)
        #print(cv_best.transpose())
        cv_best.transpose().to_csv(cv_best_path, header=True, index=True)
    else:
        best_algos = 'None of the algorithms found Seasonality trends'
        ssn_obvd = ("No seasonality trend observed in stock price of %s for prev. %d years"
                    % (current.ticker, current.cross_val))

    #print(ssn_obvd)
    LOG.info("Seasonality Predictions : %s", str(ssn_obvd))
    result = result_row(row.ListingTicker, row.ListingName
                        , seasonality=ssn_obvd
                        , models=best_algos
                        , outputModelFolder=cv_model_path
                        , outputDataFolder=','.join([cv_all_path, cv_best_path])
                        , outputImageFolder=','.join([cv_plot_all_path, cv_plot_bst_path])
                        )
    #print(str(result))
    LOG.info("Completed Seasonality cross-validation for %s", ticker)
    return result

def predict(row, is_test=False):
    """ Seasonality prediction for current year """

    #Normal flow
    current = processing_row(row.ListingTicker, row.ListingName, row.AnalysisAttempt, is_test)

    ticker = current.ticker
    LOG.info("Starting Seasonality prediction for %s", current.ticker)

    #Load data_df from shared drive
    data_df = None
    yagg_df = None
    data_df = pd.read_csv(const.MODDATA_PATH + ticker + '.csv', index_col=0, parse_dates=True)
    yagg_df = pd.read_csv(const.MODDATA_PATH + ticker + const.FILE_YR_AGG, index_col=0)
    wd_series = data_df[wd_col]

    years_of_data = len(yagg_df)
    current.is_prediction = True
    #current.max_pred_price = data_df[price_col].max() * 2
    current.max_pred_price = data_df[data_df.year >= const.CURRENT_YEAR - min_trn_yrs]\
        [price_col].max() * 2
    current.cross_val = 1 #For Prediction, no cross validations
    current.frequency = int(round(len(data_df) / years_of_data, 0))
    current.split_year = const.CURRENT_YEAR - min_trn_yrs
    #print('Average no. of records per year:', current.frequency)
    #print('Train data period:', len(yagg_df), 'years')
    #print('Seasonal Decompose begins from:', current.split_year)
    LOG.info("Average no. of records per year: %d", current.frequency)
    LOG.info("Train data period: %d years", len(yagg_df))
    LOG.info("Seasonal Decompose begins from: %d", current.split_year)

    pred_model_path = const.MODEL_PATH + ticker # + '_cv.pkl'
    pred_sd_path = const.CSV_PATH + ticker + '_pred_sd.csv'
    pred_fc_path = const.CSV_PATH + ticker + '_pred_fc.csv'
    pred_all_path = const.CSV_PATH + ticker + '_pred_all.csv'
    pred_best_path = const.CSV_PATH + ticker + '_pred_best.csv'
    pred_plot_all_path = const.IMG_PATH + ticker + '_pred_all.png'
    pred_plot_bst_path = const.IMG_PATH + ticker + '_pred_best.png'

    LOG.info("Starting Seasonality prediction for %s", current.ticker)

    #Seasonal Decompose based Prediction
    pred_sd_refined = pred_sd(current, data_df, wd_series, pred_sd_path)
    #print(pred_sd_refined)
    LOG.info("Seasonal Decompose based Predictions: \n %s", str(pred_sd_refined))

    if (not os.path.exists(pred_fc_path)) | is_test:
        #Estimate the number of differences required for Auto-ARIMA
        if current.diff is None:
            current = get_aa_diff(data_df[price_col].copy(), current)

        ts_fcast(current, data_df, None, None, pred_fc_path)

    #Read output data from csv
    pred_sd_data = pd.read_csv(pred_sd_path, index_col=0, parse_dates=True)
    #Date col = ds = 1st column
    pred_fcast = pd.read_csv(pred_fc_path, index_col=0, parse_dates=[1])

    if not os.path.exists(pred_plot_all_path):
        plot_cv(stock=current, sd_df=pred_sd_data, output_df=pred_fcast.set_index('ds')
                , file_path=pred_plot_all_path, lalgos=None)

    ts_refined = pred_fcast.groupby(pred_fcast.ds.dt.year).idxmin()
    min_wd_pred = ts_refined.drop(['ds'], axis=1).transpose()
    min_wd_pred = min_wd_pred.append(pred_sd_refined.transpose())
    min_wd_pred.columns = [pred_wd_col]
    #print(min_wd_pred)

    #Save predictions as date of Min. price
    pred_all = min_wd_pred.copy().dropna()
    pred_all[pred_dt_col] = pred_all[pred_wd_col].\
        apply(lambda x: pred_dates.loc[x].ds.strftime("%d %b"))
    pred_all[method_col] = pred_all.apply(lambda row: get_desc_from_code(row.name), axis=1)
    #print(pred_all.transpose())
    pred_all[[method_col, pred_dt_col]].to_csv(pred_all_path, header=True, index=False)

    #If there are more than 2 entries for both first month and last month, normalize the data
    if ((min_wd_pred.pred_wd_min < pred_range).sum() > 2) &\
            ((min_wd_pred.pred_wd_min > (len(pred_dates) - pred_range)).sum() > 2):
        min_wd_pred.loc[min_wd_pred.pred_wd_min < pred_range, pred_wd_col] = \
            min_wd_pred.pred_wd_min + len(pred_dates)

    output_range = align_pred_output(current, min_wd_pred.copy())
    #print(output_range)
    LOG.info("Prediction Range: %s", str(output_range))
    ssn_pred = ("Seasonality not indicated by stock for current year: %d"
                % (const.CURRENT_YEAR))
    best_algos = ""

    if not output_range.empty:
        lst_algos = list(output_range.index.values)
        best_algos = "Following algorithms gave consistent seasonality predictions"
        best_algos = best_algos + f" for the current year: {const.CURRENT_YEAR}"
        best_df = algos_df[algos_df['CodeName'].isin(lst_algos)]
        #print(best_df)
        for algo in best_df.itertuples():
            best_algos = best_algos + '\n' + algo.Description
        #print(best_algos)
        LOG.info("Consistent Prediction Algorithms : %s", str(best_algos))


        #Save best predictions as date of Min. price
        pred_best = output_range.copy().dropna()
        pred_best.loc[pred_best[pred_wd_col] > len(pred_dates), pred_wd_col] -= len(pred_dates)
        pred_best[pred_dt_col] = pred_best[pred_wd_col].\
            apply(lambda x: pred_dates.loc[x].ds.strftime("%d %b"))
        pred_best[method_col] = pred_best.apply(lambda row: get_desc_from_code(row.name), axis=1)
        #print(pred_best.transpose())
        pred_best[[method_col, pred_dt_col]].to_csv(pred_best_path, header=True, index=False)

    plot_cv(stock=current, sd_df=pred_sd_data, output_df=pred_fcast.set_index('ds')
            , file_path=pred_plot_bst_path, lalgos=lst_algos)
    if (output_range.max()[0] - output_range.min()[0]) <= pred_range:
        from_wd = int(output_range.min()[0])
        if int(output_range.min()[0]) > len(pred_dates):
            from_wd = int(output_range.min()[0] - len(pred_dates))

        to_wd = int(output_range.max()[0])
        if int(output_range.max()[0]) > len(pred_dates):
            to_wd = int(output_range.max()[0] - len(pred_dates))

        if from_wd == to_wd:
            ssn_pred = ("Predicted Working day for stock to have min. value: %s"
                        % (pred_dates.loc[from_wd, 'ds'].strftime("%d %b"))
                        )
        else:
            med_wd = int(output_range.median()[0])
            if int(output_range.median()[0]) > len(pred_dates):
                med_wd = int(output_range.median()[0] - len(pred_dates))

            msg = "Predicted Working day range for stock to have min. value:"
            ssn_pred = ("%s %s to %s with median at %s"
                        % (msg, pred_dates.loc[from_wd, 'ds'].strftime("%d %b")
                           , pred_dates.loc[to_wd, 'ds'].strftime("%d %b")
                           , pred_dates.loc[med_wd, 'ds'].strftime("%d %b"))
                        )

    #print(ssn_pred)
    LOG.info("Seasonality Predictions : %s", str(ssn_pred))

    #Analyze output of different methods
    LOG.info("Starting comparison of prediction outputs of different methods for %s"
             , current.ticker)

    LOG.info("Completed Seasonality prediction for %s", current.ticker)
    result = result_row(row.ListingTicker, row.ListingName
                        , seasonality=ssn_pred
                        , models=best_algos
                        , outputModelFolder=',' + pred_model_path
                        , outputDataFolder=',' + ','.join([pred_all_path, pred_best_path])
                        , outputImageFolder=',' + ','.join([pred_plot_all_path, pred_plot_bst_path])
                        )
    #print(str(result))
    return result
