import datetime
from datetime import date

import numpy as np
import pandas as pd
import pandas_datareader as web
from dateutil.relativedelta import relativedelta
from matplotlib.ticker import FuncFormatter
from pypfopt import (HRPOpt, discrete_allocation, expected_returns,
                     risk_models)
from pypfopt.efficient_frontier import EfficientFrontier


def last_day_month(any_day):
  next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
  return next_month - datetime.timedelta(days=next_month.day)

#Obtains price data
class Data_download_processing():
  def __init__(self,backtest_mons, test_mons, stock_tickers):
    self.backtest_mons=backtest_mons
    self.test_mons=test_mons
    if self.backtest_mons%test_mons!=0:
      raise ValueError("backtest months has to be a multiple of test months")
    self.stock_tickers = stock_tickers

  def get_stock_data(self):
    self.today_date = date.today()
    self.start_date= (self.today_date - relativedelta(months=+self.backtest_mons)).replace(day=1)
    self.start_date_str= (self.today_date - relativedelta(months=+self.backtest_mons)).replace(day=1).strftime('%Y-%m-%d')

    self.stock_tickers_clean = self.stock_tickers.split(',')
    thelen = len(self.stock_tickers_clean)
    self.price_data = []
    for ticker in range(thelen):
      self.prices = web.DataReader(self.stock_tickers_clean[ticker], start=self.start_date_str, end = self.today_date, data_source='yahoo')
      self.price_data.append(self.prices.assign(ticker=ticker)[['Adj Close']])
    self.stocks_df = pd.concat(self.price_data, axis=1)
    self.stocks_df.columns= self.stock_tickers_clean
    return self.stocks_df

#Sets backtest periods
  def set_training_period(self):
    self.training_end=(self.start_date - relativedelta(months=-(self.backtest_mons-self.test_mons))).replace(day=1)
    self.training_end= last_day_month(self.training_end)
    self.training_end_str=self.training_end.strftime('%Y-%m-%d')
    mask=(self.stocks_df.index >= self.start_date_str) & (self.stocks_df.index <= self.training_end_str)
    self.train_data_2 = self.stocks_df.loc[mask]
    return self.train_data_2

  def set_test_period(self):
    self.test_start=self.training_end + datetime.timedelta(days=2)
    self.test_start_str=self.test_start.strftime('%Y-%m-%d')
    self.test_end_str=last_day_month(self.test_start).strftime('%Y-%m-%d')
    mask=(self.stocks_df.index >= self.test_start_str) & (self.stocks_df.index <= self.test_end_str)
    self.test_data_2 = self.stocks_df.loc[mask]
    return self.test_data_2

class Optimise_model():
  def __init__(self, optimiser,stocks_data_frame, aum):
    self.optimiser=optimiser
    self.stocks_data_frame=stocks_data_frame
    self.aum=aum

  def mvo_optimiser(self):
    #Expected returns
    mu = expected_returns.mean_historical_return(self.stocks_data_frame)
    #Covariance matrix for the prices and tickers
    sigma = risk_models.sample_cov(self.stocks_data_frame)
    #Limits to long-only
    ef = EfficientFrontier(mu, sigma, weight_bounds=(0,1))
    #Builds weights for minimum volatility
    weights = ef.min_volatility()
    cleaned_weights = ef.clean_weights()
    mvo_performance = ef.portfolio_performance()
    latest_prices = discrete_allocation.get_latest_prices(self.stocks_data_frame)
    allocation_maxs, rem_maxs = discrete_allocation.DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=self.aum).lp_portfolio()
    return cleaned_weights, mvo_performance, allocation_maxs

  def msr_optimiser(self):
    mu = expected_returns.mean_historical_return(self.stocks_data_frame)
    sigma = risk_models.sample_cov(self.stocks_data_frame)
    ef = EfficientFrontier(mu, sigma, weight_bounds=(0,1))
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    msr_performance = ef.portfolio_performance()
    latest_prices = discrete_allocation.get_latest_prices(self.stocks_data_frame)
    allocation_maxs, rem_maxs = discrete_allocation.DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=self.aum).lp_portfolio()
    return cleaned_weights, msr_performance, allocation_maxs

  def hrp_optimiser(self):
    #Hierarchical risk parity uses daily percentage changes to cluster and determine optimal allocation
    returns = self.stocks_data_frame.pct_change().dropna(how="all")
    hrp=HRPOpt(returns)
    hrp.optimize()
    cleaned_weights = hrp.clean_weights()
    hrp_performance=hrp.portfolio_performance()
    latest_prices = discrete_allocation.get_latest_prices(self.stocks_data_frame)
    allocation_maxs, rem_maxs = discrete_allocation.DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=self.aum).lp_portfolio()
    return cleaned_weights, hrp_performance, allocation_maxs

  def optimise_portfolio(self):
    if self.optimiser=="mvo":
      weights, performance, allocation_maxs=self.mvo_optimiser()
    if self.optimiser=="msr":
      weights, performance, allocation_maxs=self.msr_optimiser()
    if self.optimiser=="hrp":
      weights, performance, allocation_maxs=self.hrp_optimiser()
    return weights, performance, allocation_maxs

#Walk forward backtesting
class Backtest_stats():
  def __init__ (self, stocks_dataframe_to_backtest, portfolio_weights, test_months):
    self.stocks_dataframe_to_backtest=stocks_dataframe_to_backtest
    self.portfolio_weights=portfolio_weights
    self.test_months=test_months

  def get_annualized_returns(self):
    self.daily_returns = self.stocks_dataframe_to_backtest.iloc[0:].pct_change()*100
    self.avg_daily_returns = self.daily_returns.mean()
    portfolio_return = []
    weightage = list(self.portfolio_weights.values())
    for i in range(len(self.avg_daily_returns)):
      portfolio_return.append(weightage[i]*self.avg_daily_returns[i])
    self.annualized_returns=(1+sum(portfolio_return))**(12/self.test_months)-1
    return self.annualized_returns

  def get_annualized_volatility(self):
    self.daily_std = self.daily_returns.std()
    portfolio_std = []
    weightage = list(self.portfolio_weights.values())
    for i in range(len(self.daily_std)):
      portfolio_std.append(weightage[i]*self.daily_std[i])
    annualized_std = sum(portfolio_std) * 252 ** 0.5
    return annualized_std

  def get_sharpe_ratio(self):
    self.sharpe_ratio = (self.avg_daily_returns-(0.01))/self.daily_std
    portfolio_sharpe = []
    weightage = list(self.portfolio_weights.values())
    for i in range(len(self.sharpe_ratio)):
      portfolio_sharpe.append(weightage[i]*self.sharpe_ratio[i])
    portfolio_sharpe=sum(portfolio_sharpe)
    return portfolio_sharpe

