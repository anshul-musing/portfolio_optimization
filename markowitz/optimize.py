import pandas as pd
import numpy as np
from pulp import *


def opt_model(assets, scenarios, net_return, max_risk, min_assets):

    # Define problem
    invest = LpProblem("Investment allocation problem", LpMaximize)

    # Variables
    y = LpVariable.dicts("Choose asset", assets, lowBound=0.0, upBound=1.0, cat=LpInteger)
    w = LpVariable.dicts("Allocation fraction", assets, lowBound=0.0, upBound=1.0, cat=LpContinuous)
    z = LpVariable.dicts("Risk measure", scenarios, lowBound=0.0, upBound=None, cat=LpContinuous)

    # Objective function
    # maximize return across all assets in all scenarios
    invest += lpSum(w[i]*net_return[s,i] for i in assets for s in scenarios) / len(scenarios)

    # Allocation fraction sum to 1
    invest += lpSum(w[i] for i in assets) == 1.0

    # Allocate asset if chosen, with a
    # minimum allocation of 1%
    for i in assets:
        invest += w[i] <= y[i]
        invest += y[i] <= 100*w[i]

    # Minimum number of assets in the portfolio
    invest += lpSum(y[i] for i in assets) >= min_assets

    # Risk of negative returns
    for s in scenarios:
        invest += z[s] >= -lpSum(w[i]*net_return[s,i] for i in assets)

    # Risk tolerance
    invest += lpSum(z[s] for s in scenarios) / len(scenarios) <= max_risk

    # Solve the model
    invest.solve(PULP_CBC_CMD(msg=1))
    opt_obj = value(invest.objective)*100.0
    opt_w = {}
    for i in assets:
        opt_w[i] = w[i].varValue

    opt_risk = sum([z[s].varValue for s in scenarios]) / len(scenarios)

    return opt_obj, opt_w, opt_risk


def optimize(rdf, max_risk, min_assets):

    # Define problem parameters
    assets = [i for i in range(len(rdf.columns))]
    scenarios = rdf.index.tolist()
    net_return = rdf.to_numpy()

    # Solve the problem
    return opt_model(assets, scenarios, net_return, max_risk, min_assets)


def perform_opt(rdf, exp_ratio, min_ratio=0.3):
    min_assets = 1 # minimum assets in the portfolio
    max_risk = [0.14, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02] # max risk of negative returns
    res_list = []
    for m in max_risk:
        # Get the daily equivalent of the annual max risk
        mrisk = (1+m)**(1/360.0)-1

        # Filter out high expense ratio assets
        exp_r_filtered = {k:v for (k,v) in exp_ratio.items() if v <= 0.4 and k in rdf.columns}
        rdff = rdf[list(exp_r_filtered.keys())].copy()

        # Optimize
        obj, wopt, orisk = optimize(rdff, mrisk, min_assets)
        
        # Save results
        wopt.update({'return':obj, 'risk': orisk})
        if obj > 0:
            res_list.append(wopt)

    resdf = pd.DataFrame(res_list)

    # Attach ticker symbols as column names
    col_list = {}
    for i, c in enumerate(rdff.columns):
        col_list[i] = c
    resdf = resdf.rename(columns=col_list)

    # Remove the zero columns
    collist = []
    for c in resdf.columns:
        if resdf[c].sum() > 0:
            collist.append(c)

    return resdf[collist]
