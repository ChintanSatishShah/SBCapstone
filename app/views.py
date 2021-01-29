"""This is where application routes are defined"""
from os import path
import sys
import logging
from flask import render_template, request, jsonify, send_from_directory
from wtforms import StringField, Form
from wtforms.validators import DataRequired, Length
from werkzeug.exceptions import BadRequest #, HTTPException
import pandas as pd

from . import app#, APP_ROOT
from .models import get_forecast_status, get_stock_list
from .constants import ROOT, CSV_PATH, IMG_PATH, MAX_FETCH_ATTEMPTS, DATA_SFX
from .constants import CV_ALL_SFX, CV_PLT_ALL_SFX, CV_PLT_BST_SFX   #, CV_BST_SFX
from .constants import PRED_ALL_SFX, PRED_PLT_ALL_SFX, PRED_PLT_BST_SFX, PRED_BST_SFX
# pylint: disable=invalid-name

LOG = logging.getLogger(__name__)

#Initialize a global variable to hold the Stock Name and Ticker as dictionary
res = get_stock_list()
dict_stock = {res[i].ListingTicker : res[i].ListingName for i in range(0, len(res))}

def set_color(content):
    """ Colors cells in a dateframe based on value """
    style = ''
    if '~' in content:
        pctile = float(content.split('~')[1])
        if pctile <= 20:
            style = 'color: limegreen; font-weight: bold'
        #elif pctile <= 50:
            #style = 'yellow'
        #else: #pctile > 50
            #style = 'lightcoral'

    #return 'background-color: %s' % color
    return style

class SearchForm(Form):
    """ Main class of Form """
    stockpicker = StringField(
        label='Enter Stock Name:'
        , validators=[DataRequired(), Length(max=100)]
        , render_kw={"placeholder" : "Enter the Stock name"}
    )

@app.route("/home")
@app.route("/index")
@app.route('/', methods=['GET', 'POST'])
def index():
    """ Home page """
    form = SearchForm(request.form)
    return render_template("base.html", form=form)

@app.route('/favicon.ico')
def favicon():
    """ Favicon """
    return send_from_directory(r'static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/stocks')
def stock_dict():
    """ Set Stock Data for Search """
    list_stocks = [r.as_dict() for r in res]
    return jsonify(list_stocks)

@app.route('/process', methods=['POST'])
def process():
    """ Fill Stock Data """
    msg = ""
    display_html = ""

    if bool(request.form) & ('stock' in request.form):
        form_value = request.form['stock']

        try:
            if " " in form_value:
                #Check for Stock Name
                sel_ticker = list(dict_stock.keys())[list(dict_stock.values()).index(form_value)]
            else:
                form_value = form_value.upper()
                if form_value in dict_stock:
                    sel_ticker = form_value
                else:
                    msg = f'No stock exists by name/ticker: '
                    msg = msg + f'<b style="color:red;">{form_value}</b>'
                    display_html = f'<p>{msg}. Please enter and select a valid stock.</p>'
                    return jsonify({'error': display_html})
        except (ValueError) as err:
            LOG.info("No stock exists by the name: %s", form_value)
            LOG.error("\nException Line No.: %s  \nMsg: %s"
                      , str(sys.exc_info()[2].tb_lineno), str(err))
            msg = f'No stock exists by the name: <b style="color:red;">{form_value}</b>'
            display_html = f'<p>{msg}. Please enter and select a valid stock.</p>'
            return jsonify({'error': display_html})
        #except (TypeError, HTTPException) as err:
            #LOG.info("Error occurred while fetching data for stock: %s", form_value)
            #LOG.error("\nException Line No.: %s  \nMsg: %s"
                      #, str(sys.exc_info()[2].tb_lineno), str(err))
            #msg = f'Error occurred while fetching data for stock: '
            #msg = msg + '<b style="color:red;">{form_value}</b>'
            #display_html = f'<p>{msg}. Please enter and select a valid stock.</p>'
            #return jsonify({'error': display_html})
    else:
        #throws Bad Request Exception
        LOG.info("Incorrect request format: %s", str(request.form))
        raise BadRequest

    #Code to fetch status of stock from DB
    row = get_forecast_status(sel_ticker)

    if bool(row):
        ticker = row.ListingTicker
        LOG.info("Started UI rendering for stock: %s", ticker)

        msg = f'Seasonality analysis for <b>{ticker}</b>'
        display_html = f'<p style="color:blue;">{msg}</p>'
        data_path = ""
        img_path = ""
        res_price = ""
        res_past = ""
        res_pred = ""

        utflag = ''
        if 'ut' in request.form:
            utflag = request.form['ut']

        try:
            if (row.RawDataFilePath is None) | (utflag.startswith('Not')):
                if (row.DataCollectionAttempt < MAX_FETCH_ATTEMPTS) | (utflag == 'NotFetched'):
                    msg = ("Data is not yet fetched for stock: %s" % (ticker))
                    LOG.info(msg)
                    display_html += f'<p style="color:darkorange;">{msg}</p>'
                else: #row.DataCollectionAttempt >= MAX_FETCH_ATTEMPTS
                    msg = ("Data is not available for stock: %s" % (ticker))
                    LOG.info(msg)
                    display_html += f'<p style="color:red;">{msg}</p>'
            else:
                if (row.CrossValidationCount == 0) | (utflag.startswith('Raw')):
                    msg = ("Data is not yet cleaned for stock: %s" % (ticker))
                    LOG.info(msg)
                    display_html += f'<p style="color:darkorange;">{msg}</p>'
                else:
                    if row.InputDataFilePath is None:
                        msg = ("Less than 1 year of historical data avl. for stock: %s" % (ticker))
                        LOG.info(msg)
                        display_html += f'<p style="color:darkorange;">{msg}</p>'
                    else:
                        img_path = IMG_PATH + ticker + DATA_SFX

                        #img_data = f'<p style="color:red;">File not found: {img_path}</p>'
                        #if path.exists(ROOT + img_path):
                        msg = f"Historical Stock Price data for {ticker}"
                        img_data = f'<img src="{img_path}" alt="{msg}" class="plotd"></img>'

                        #display_html += img_data
                        res_price = img_data

                        if row.CrossValidationCount < 0: #identify suitable stock for ut
                            msg = "Data is insufficient for Seasonality analysis " \
                                + ("of stock: %s" % (ticker))
                            LOG.info(msg)
                            display_html += f'<p style="color:darkorange;">{msg}</p>'
                        else: #row.CrossValidationCount > 0:
                            if (row.SeasonalityObserved is None) | (utflag.startswith('Pending')):
                                msg = ("Seasonality analysis is pending for stock: %s" % (ticker))
                                LOG.info(msg)
                                display_html += f'<p style="color:darkorange;">{msg}</p>'
                            else:
                                #CV completed
                                msg = row.SeasonalityObserved
                                LOG.info(msg)

                                if (row.BestModels is None) | (utflag.startswith('All')):
                                    #No seasonality seen for prev. years - Show all plots and data
                                    res_past = f'<p style="color:red;">{msg}</p>'
                                    data_path = CSV_PATH + ticker + CV_ALL_SFX
                                    style = 'all'
                                    img_path = IMG_PATH + ticker + CV_PLT_ALL_SFX
                                else:
                                    LOG.info("Best Models: %s", row.BestModels)
                                    #Seasonality seen for prev. years - Show best plots and data
                                    #data_path = CSV_PATH + ticker + CV_BST_SFX
                                    #Seasonality seen for CV years - Show all plots and data
                                    res_past = f'<p style="color:blue;">{msg}</p>'
                                    data_path = CSV_PATH + ticker + CV_ALL_SFX #CV_BST_SFX
                                    style = 'best'
                                    img_path = IMG_PATH + ticker + CV_PLT_BST_SFX

                                tbl_cv = f'<p style="color:red;">File not found: {data_path}</p>'
                                if path.exists(data_path):
                                    cv_df = pd.read_csv(data_path).fillna(value='NA')
                                    #cv_df.index += 1
                                    header = 'Algorithm - Pred. Min. Price Date'
                                    cv_df = cv_df.rename(columns={cv_df.columns[0]: header})

                                    if utflag:
                                        #Special handling to highlight the 1st row using class = cv
                                        tbl_cv = cv_df.to_html(classes=style + ' cv')
                                    else:
                                        tbl_cv = cv_df.style\
                                            .applymap(set_color
                                                      , subset=list(cv_df.columns[1:].values))\
                                            .set_table_attributes(f'class="dataframe {style} cv"')\
                                            .format(lambda x: x.split('~')[0]).render()

                                    LOG.info("ct_table: %s", str(tbl_cv))

                                #img_cv = f'<p style="color:red;">File not found: {img_path}</p>'
                                #if path.exists(ROOT + img_path):
                                alt = f'alt="Past analysis for {ticker}"'
                                img_cv = f'<img src="{img_path}" {alt} class="ploto"></img>'

                                #display_html = display_html + tbl_cv + img_cv
                                display_html += res_past
                                res_past = res_past + tbl_cv + img_cv

                            if (row.SeasonalityPredicted is None) | (utflag.startswith('Pending')):
                                msg = ("Time Series prediction pending for stock: %s" % (ticker))
                                LOG.info(msg)
                                display_html += f'<p style="color:darkorange;">{msg}</p>'
                            else:
                                #Prediction completed
                                msg = row.SeasonalityPredicted
                                LOG.info(msg)

                                if (row.ConsistentModels is None) | (utflag.startswith('All')):
                                    #No/seasonality Predicted - Display all plots and data
                                    msg_html = f'<p style="color:red;">{msg}</p>'
                                    prd_path = CSV_PATH + ticker + PRED_ALL_SFX
                                    style = 'all'
                                    img_path = IMG_PATH + ticker + PRED_PLT_ALL_SFX
                                else:
                                    LOG.info("Consistent Models: %s", row.ConsistentModels)
                                    #Seasonality Predicted - Display best plots and data
                                    if ':' in msg:
                                        parts = msg.split(':')
                                        reco = "Suggested best period for investment:"
                                        msg_html = f'<p style="color:blue;">{reco}\
                                            <b style="background-color:yellow;">{parts[1]}</b>\
                                            </p>'
                                    else:
                                        msg_html = f'<p style="color:blue;">{msg}</p>'

                                    prd_path = CSV_PATH + ticker + PRED_BST_SFX
                                    style = 'best'
                                    img_path = IMG_PATH + ticker + PRED_PLT_BST_SFX

                                tbl_pred = f'<p style="color:red;">File not found: {prd_path}</p>'
                                if path.exists(prd_path):
                                    #Order of columns in csv will change the logic
                                    prd_df = pd.read_csv(prd_path) #, usecols=[2, 3]
                                    prd_df.index += 1
                                    #tbl_pred = prd_df.iloc[:, [1, 0]].to_html(classes=style)
                                    tbl_pred = prd_df.to_html(classes=style)

                                #img_pred = f'<p style="color:red;">File not found: {img_path}</p>'
                                #if path.exists(ROOT + img_path):
                                alt_pred = f'alt="Prediction for {ticker}"'
                                img_pred = f'<img src="{img_path}" {alt_pred} ' +\
                                        f'class="ploto"></img>'

                                #display_html = display_html + tbl_pred + img_pred
                                display_html += msg_html
                                res_pred = msg_html + tbl_pred + img_pred

            #data = pd.read_csv(os.path.join(APP_ROOT, r'./output/csv/3MINDIA_pred_best.csv'))
            return jsonify({'overall': display_html
                                       , 'price': res_price
                                                  , 'past': res_past
                                                            , 'pred': res_pred})

        except (TypeError, ValueError, AttributeError) as err:
            LOG.info("Error occurred while fetching data for stock: %s", form_value)
            LOG.error("\nException Line No.: %s  \nMsg: %s"
                      , str(sys.exc_info()[2].tb_lineno), str(err))
            msg = f'<p>Error occurred while fetching data for stock: '
            msg = msg + f'<b style="color:red;">{form_value}</b>' + '.</p>'
            display_html = display_html + msg
            return jsonify({'error': display_html})

    else:
        LOG.info("Improbable scenario - selected stock: %s", form_value)
        return jsonify({'error': f'<p style="color:red;">Improbable scenario. ' \
                                 + f'How did you reach here?</p>'})
    