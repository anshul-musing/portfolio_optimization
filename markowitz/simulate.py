import pandas as pd
import numpy as np


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
            returns[t].append(df.iloc[idx][t+'-ret'].prod() - 1 - (exp_ratio[t]*ndays/(365.0*100)))

    rdf = pd.DataFrame(returns)

    return rdf


def portfolio_return(tickers, rdf, w):

    # Create a portfolio return as a weighted sum of
    # the returns of the investments in the portfolio
    # for a given set of weights
    rdf = rdf.assign(portfolio_return=0.0)
    for i, t in enumerate(tickers):
        rdf['portfolio_return'] += w[i]*rdf[t]

    # number of scenarios with negative returns, average negative
    # return across scenarios, and average portfolio return
    perc_neg_scenarios = rdf[rdf.portfolio_return < 0].shape[0] / rdf.shape[0]
    avg_neg_return = rdf[rdf.portfolio_return < 0]['portfolio_return'].sum() / rdf.shape[0] 
    avg_return = rdf['portfolio_return'].mean()
    
    return rdf, perc_neg_scenarios, avg_neg_return, avg_return


def efficient_frontier(tickers, rdf):

    # Randomly sample weights for the investments in
    # the portfolio and create an efficient frontier
    # between average portfolio return and negative
    # portfolio return for given weights
    np.random.seed(707)
    avgret = []
    negret = []
    for r in range(1000):
        w = np.random.random(len(tickers))
        w = w/sum(w)
        _, _, avg_neg, avg_ret = portfolio_return(tickers, rdf, w)
        negret.append(-avg_neg)
        avgret.append(avg_ret)

    # plot efficient frontier as
    # plt.scatter(negret, avgret)
    return negret, avgret

