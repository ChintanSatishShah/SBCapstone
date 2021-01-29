""" Plots for analyzer """

#import pandas as pd
import matplotlib.pyplot as plt

import analyzer.constants as const
from analyzer.constants import price_col, price_sma, pctile_col
from analyzer.constants import algos_df, min_trn_yrs

#from analyzer.forecast import processing_row

def plot_data(sticker, data_df, yagg_df):
    """ Plots Historical stock price data for each stock """
    fig, axes = plt.subplots(nrows=3, ncols=1, squeeze=False, figsize=(16, 11)
                             , gridspec_kw={'height_ratios': [4, 3, 4]})
    fig.suptitle(f"{sticker} - Historical stock price data", fontsize='x-large', fontweight='bold')

    fig1 = axes[0][0]
    fig1.plot(data_df[[price_col, price_sma]])
    fig1.set_ylabel('Price')
    #fig1.set_xticks(data_df.groupby(year_col)[wd_col].idxmin())
    #fig1.set_xticklabels([])
    fig1.legend(['Adjusted Close', 'Adjusted Close (SMA)'], loc='best')
    fig1.set_title('Comparison of adjusted close and it\'s 5 day SMA')

    fig2 = axes[1][0]
    fig2.plot(data_df[[pctile_col]])
    fig2.set_ylabel('Percentile')
    #fig2.sharex(fig1)
    #fig2.set_xticklabels([])
    fig2.legend([r'Adjusted Close %ile'], loc='best')
    fig2.set_title('Adjusted close price perecentile values grouped on yearly basis')

    fig3 = axes[2][0]
    fig3.plot(yagg_df[['doy_min_adj_close', 'doy_min_sma']])
    fig3.set_ylabel('Day of year')
    fig3.set_xticks(yagg_df.index.values)
    fig3.legend(['Min. Adjusted Close', 'Min. Adjusted Close (SMA)'], loc='best')
    fig3.set_title('Comparison of Day of year with min. adjusted close and it\'s 5 day SMA')

    for axis in fig.axes:
        #axis.tick_params(labelrotation=45)
        axis.grid()

    plt.tight_layout()
    plt.savefig(const.IMG_PATH + sticker + const.IMG_DATA) #, bbox_inches="tight"
    #plt.show()
    plt.close(fig)

def plot_cv(stock, sd_df, output_df, file_path, lalgos):
    """ Plots Cross Validation output data for each stock """
    fig, axes = plt.subplots(nrows=3, ncols=2, squeeze=False, figsize=(16, 11)
                             , gridspec_kw={'height_ratios': [4, 3, 4]})
    isbest = "(All)" if lalgos is None else "(Best)"
    if stock.is_prediction:
        fig.suptitle(r"$\bf{" + stock.ticker + r"\ -\ Current\ year\ Seasonality\ prediction\ "
                     + isbest + "}$" + "\n" + f"Seasonal Decompose for prev. {min_trn_yrs} years + "
                     + f"Time-series forecast price for current year"
                     , fontsize='x-large')
    else:
        sd_years = stock.cross_val + min_trn_yrs - 1
        fig.suptitle(r"$\bf{" + stock.ticker + r"\ -\ Past\ Seasonality\ Analysis\ "
                     + isbest + "}$" + "\n" + f"Seasonal Decompose for prev. {sd_years} years + "
                     + f"Comparison of actual vs forecast price for prev. {stock.cross_val} years"
                     , fontsize='x-large')#, fontweight='bold'

    for row in range(3):
        for col in range(2):
            sub = axes[row][col]
            plt_id = (col * 3) + row + 1
            algos = algos_df[(algos_df.PlotId == plt_id)]

            #Set subplot Title before filtering algos
            sub.set_title(algos.iloc[0].Method)

            #Filter for best
            if lalgos is not None:
                algos = algos[(algos.CodeName.isin(lalgos))]
            else:
                algos = algos[(algos.CodeName.isin(list(sd_df.columns.values)
                                                   + list(output_df.columns.values)))]

            #Plot if Algo data exists in input DF
            if not algos.empty:
                sub.grid() #Enable grids
                if col > 0:
                    if stock.is_prediction:
                        sub.set_prop_cycle('color', list(algos.Color.values))
                        plot_df = output_df[list(algos.CodeName.values)]
                        sub.plot(plot_df)
                        sub.legend(list(algos.DisplayName.values), loc='upper left')

                    else:
                        sub.set_prop_cycle(color=['tab:green'] + list(algos.Color.values))
                        plot_df = output_df[[price_col] + list(algos.CodeName.values)]
                        sub.plot(plot_df)
                        sub.legend(['Actual Adjusted Close Price'] + list(algos.DisplayName.values)
                                   , loc='upper left')

                    sub.set_ylabel('Predicted Stock Price')
                    sub.set_ylim(max(0, plot_df.min().min())
                                 , min(stock.max_pred_price, plot_df.max().max()))

                else:
                    sub.set_prop_cycle(color=list(algos.Color.values))
                    if row < 2:
                        sub.plot(sd_df[list(algos.CodeName.values)])
                    else:
                        sub.plot(output_df[list(algos.CodeName.values)])
                    sub.legend(list(algos.DisplayName.values), loc='upper left')
                    sub.set_ylabel('Price Trends')
                    sub.set_yticklabels([])

                #sub.set_xticks(output_df.groupby(output_df.index.year)[wd_col].idxmin())
                #sub.set_xticklabels([])
            else:
                sub.annotate('No seasonality observed', xy=(0.02, 0.9), xycoords="axes fraction")
                sub.set_xticklabels([])
                sub.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(file_path) #, bbox_inches="tight"
    #plt.show()
    plt.close(fig)
