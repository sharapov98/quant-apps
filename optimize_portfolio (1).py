#Abdul Sharopov and Abhaya Gauchan

import argparse
from pypfopt import plotting
import matplotlib.pyplot as plt
import optimize_class
from optimize_class import (Backtest_stats, Data_download_processing,
                            Optimise_model)

parser = argparse.ArgumentParser(description='optimizer')
def enable_parsing(parser):

    parser.add_argument("--tickers", help="Stock tickers",
                        required=True)
    parser.add_argument("--optimizer", help="type",
                        required=True)
    parser.add_argument("--aum", type=int, help="Invested amount")
    parser.add_argument("--backtest_months",
                        type=int, required=True)
    parser.add_argument("--test_months",
                        type=int, required=True)
    parser.add_argument("--backtest_type",
                        required=True)
    parser.add_argument('--plot_weights', help="To plot",
                        action='store_true')
    return parser

parser = enable_parsing(parser)
args = parser.parse_args()

TICKER = args.tickers
OPT = args.optimizer
AUM = args.aum
B_MONTHS = args.backtest_months
T_MONTHS = args.test_months
PLOTTER = args.plot_weights

#Check for correct number of backtest months
if B_MONTHS % T_MONTHS !=0:
  raise ValueError('Backtest_months is NOT a multiple of test_months')

#Download data for the period
data_ch2=Data_download_processing(B_MONTHS,T_MONTHS,TICKER)
data_ch2.get_stock_data()
data_ch_train=data_ch2.set_training_period()
data_ch_test=data_ch2.set_test_period()

#Optimise and output the weights and performance for the AUM
optimise_obj=Optimise_model(OPT, data_ch_train,AUM)
weights,performance,stocks_needed=optimise_obj.optimise_portfolio()

print("Expected annual returns", performance[0]*100, "%")
print("Expected annual volatility", performance[1]*100,"%")
print("Expected sharpe ratio", performance[2])

#Backtest returns
backtest_obj=Backtest_stats(data_ch_test, weights, T_MONTHS)
returns= backtest_obj.get_annualized_returns()
volatility= backtest_obj.get_annualized_volatility()
sharpe= backtest_obj.get_sharpe_ratio()
print("Realised Annualised return: ", returns,"%")
print("Realised Annual Volatility: ",volatility, "%")
print("Realised Sharpe Ratio:", sharpe)

#How many of each ticker to purchase
stocks_needed.keys()
print("Stocks needed:")
for key, value in stocks_needed.items():
  print(key, value)

if PLOTTER==True:
  plotting.plot_weights(weights)
  plt.show()
