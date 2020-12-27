import pandas as pd
import numpy as np
import yfinance as yf
from pulp import *


def ingest(tickers, start_date, end_date):

    # Download data from Yahoo Finance library 
    data = yf.download(' '.join(tickers), start=start_date, end=end_date)
    
    # Flatten the multi-index
    data.columns = [' '.join(col).strip() for col in data.columns.values]

    return data


def preprocess(data, tickers):

    # Get the relevant columns
    cols = ['Open ' + t for t in tickers]
    df = data[cols].copy().reset_index()

    # Rename columns and get daily percent return
    for t in tickers:
        df = df.rename(columns={'Open '+t:t})
        df[t+'-ret'] = 1 + (df[t].shift(-1) - df[t])/df[t]

    df = df[:-1] #last row is NaN due to differencing

    return df


def scenarios(df, tickers, exp_ratio, nscenarios, ndays):

    # Create an empty list to store yearly returns 
    # for each vehicle in each scenario
    np.random.seed(707)
    returns = {}
    for t in tickers:
        returns[t] = []

    # Simulate each scenario
    for i in range(nscenarios):
        idx = np.random.randint(0, df.shape[0], ndays)
        for t in tickers:
            returns[t].append(df.iloc[idx][t+'-ret'].prod() - 1 - (exp_ratio[t]*ndays/(360.0*100)))

    rdf = pd.DataFrame(returns)

    return rdf


def portfolio_return(tickers, rdf, w):
    rdf = rdf.assign(portfolio_return=0.0)
    for i, t in enumerate(tickers):
        rdf['portfolio_return'] += w[i]*rdf[t]

    perc_neg_scenarios = rdf[rdf.portfolio_return < 0].shape[0] / rdf.shape[0]
    avg_neg_return = rdf[rdf.portfolio_return < 0]['portfolio_return'].sum() / rdf.shape[0] 
    avg_return = rdf['portfolio_return'].mean()
    
    return rdf, perc_neg_scenarios, avg_neg_return, avg_return


def portfolio_model():
