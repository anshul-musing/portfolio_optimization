import pandas as pd
import numpy as np
from pulp import *


def opt_model(assets, scenarios, net_return, maxRisk):

    # Define problem
    invest = LpProblem("Investment allocation problem", LpMaximize)

    # Variables
    w = LpVariable.dicts("Allocation fraction", assets, lowBound=0.0, upBound=1.0, cat=LpContinuous)
    z = LpVariable.dicts("Risk measure", scenarios, lowBound=0, upBound=None, cat=LpContinuous)

    # Objective function
    # maximize return across all assets in all scenarios
    invest += lpSum(w[i]*net_return[s,i] for i in assets for s in scenarios) / len(scenarios)

    # Allocation fraction sum to 1
    invest += lpSum(w[i] for i in assets) == 1.0

    # Risk of negative returns
    for s in scenarios:
        invest += z[s] >= -lpSum(w[i]*net_return[s,i] for i in assets)

    # Risk tolerance
    invest += lpSum(z[s] for s in scenarios) / len(scenarios) <= maxRisk

    # Solve the model
    invest.solve(PULP_CBC_CMD(msg=1))
    optObj = value(invest.objective)*100.0
    optSoln = {}
    for i in assets:
        optSoln[i] = w[i].varValue    

    return optObj, optSoln


def optimize(rdf, maxRisk):

    # Define problem parameters
    assets = [i for i in range(len(rdf.columns))]
    scenarios = rdf.index.tolist()
    net_return = rdf.to_numpy()

    # Solve the problem
    return opt_model(assets, scenarios, net_return, maxRisk)
    
    return optObj, optSoln
