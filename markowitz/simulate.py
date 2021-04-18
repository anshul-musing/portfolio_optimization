import pandas as pd
import numpy as np


def scenarios(df, tickers, ndays, nscenarios=500):

    # Create an empty list to store yearly returns 
    # for each asset in each scenario
    np.random.seed(707)
    returns = {}
    for t in tickers:
        if t in df.columns:
            returns[t] = []

    # Simulate each scenario
    for i in range(nscenarios):

        # First randomly sample days from the last few years
        # of data.  We sample ndays worth of data
        idx = np.random.randint(0, df.shape[0], ndays)

        # Now we find the cumulative return for each ticker
        # on the sampled days.  Note that this methodology
        # preserves the correlation between the tickers.
        # Because each ticker's returns are calculated on the
        # same sampled set of days, we inherently get the 
        # cumulative correlated returns.  For instance, all
        # risky assets and conservative assets will show 
        # similar correlated pattern on a particular day.
        # Thus, by picking a particular day, we pick the
        # behavior of related assets on that day
        for t in tickers:
            if t in df.columns:
                ret = (df.iloc[idx][t+'-ret'].prod())**(1/float(ndays)) - 1
                returns[t].append(ret)

    rdf = pd.DataFrame(returns)

    return rdf


def portfolio_return(rdf, w):

    # Create a portfolio return as a weighted sum of
    # the returns of the assets in the portfolio
    # for a given set of weights
    pdf = rdf.copy()
    pdf = pdf.assign(portfolio_return=0.0)
    for t in rdf.columns.tolist():
        pdf['portfolio_return'] += w[t]*pdf[t]

    # number of scenarios with negative returns, average negative
    # return across scenarios, and average portfolio return
    perc_neg_scenarios = pdf[pdf.portfolio_return < 0].shape[0] / pdf.shape[0]
    avg_neg_return = pdf[pdf.portfolio_return < 0]['portfolio_return'].sum() / pdf.shape[0] 
    avg_return = pdf['portfolio_return'].mean()
    
    return pdf, perc_neg_scenarios, avg_neg_return, avg_return


def efficient_frontier(rdf):

    # Randomly sample weights for the assets in
    # the portfolio and create an efficient frontier
    # between average portfolio return and negative
    # portfolio return for given weights
    np.random.seed(707)
    avgret = []
    negret = []
    for r in range(1000):
        w = np.random.random(rdf.shape[1])
        w = w/sum(w)
        _, _, avg_neg, avg_ret = portfolio_return(rdf, w)
        negret.append(-avg_neg)
        avgret.append(avg_ret)

    # plot efficient frontier as
    # plt.scatter(negret, avgret)
    return negret, avgret

