import pandas as pd
import numpy as np
from pulp import *


def opt_model(assets, scenarios, net_return, expense_ratio
            , max_risk, min_ratio, min_assets):

    # Define problem
    invest = LpProblem("Investment allocation problem", LpMaximize)

    # Variables
    y = LpVariable.dicts("Choose asset", assets, lowBound=0.0, upBound=1.0, cat=LpInteger)
    w = LpVariable.dicts("Allocation fraction", assets, lowBound=0.0, upBound=1.0, cat=LpContinuous)
    z = LpVariable.dicts("Risk measure", scenarios, lowBound=0, upBound=None, cat=LpContinuous)

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

    # Limit on expense ratio
    if len(expense_ratio) > 0:
        for i in assets:
            invest += y[i]*expense_ratio[i] <= min_ratio

    # Solve the model
    invest.solve(PULP_CBC_CMD(msg=1))
    opt_obj = value(invest.objective)*100.0
    opt_w = {}
    for i in assets:
        opt_w[i] = w[i].varValue

    opt_risk = sum([z[s].varValue for s in scenarios]) / len(scenarios)

    return opt_obj, opt_w, opt_risk


def optimize(rdf, exp_ratio, max_risk, min_ratio, min_assets):

    # Define problem parameters
    assets = [i for i in range(len(rdf.columns))]
    scenarios = rdf.index.tolist()
    net_return = rdf.to_numpy()
    expense_ratio = list(exp_ratio.values()) if len(exp_ratio) > 0 else []

    # Solve the problem
    return opt_model(assets, scenarios, net_return, expense_ratio, 
                        max_risk, min_ratio, min_assets)
