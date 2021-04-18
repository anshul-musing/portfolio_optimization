import pandas as pd
import numpy as np
import yfinance as yf


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
    # We analyzed the daily return and found no
    # correlation with time.  In other words, the 
    # daily return appears to be independent of the
    # timeframe and we don't see any upward or
    # downward trend with time (which is different
    # from upward trend in prices over time)
    for t in tickers:
        df = df.rename(columns={'Open '+t:t})
        df[t+'-ret'] = 1 + (df[t].shift(-1) - df[t])/df[t]

    df = df[:-1] #last row is NaN due to differencing

    return df
