import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import collections
from scipy.optimize import curve_fit, minimize_scalar
from . import cons_df, m_df
from .mandatory import question_1

np.random.seed(4100142)
SALESFORCE_COST_PER_PERSON = 0.057


def utility_calc(x, A):
    return x['mean'] - 0.5*A*x['stddev']

def optional_part_1():
    range_list = np.linspace(100, 440, 11)
    original_sf = m_df['Naprosyn']['Current/Original Salesforce']
    original_revenue = m_df['Naprosyn']['Current/Original Revenue']

    cov_result = question_1(True)
    opt = cov_result['Naprosyn']['popt']
    cov = cov_result['Naprosyn']['pcov']

    def naprosyn_adbudg_util(x, params):
        mn, mx, c, d = params
        response = mn + (mx - mn) * (x**c) / (d + x**c)
        new_revenue = response*original_revenue
        new_profit = new_revenue*0.7 -  x*SALESFORCE_COST_PER_PERSON
        return new_profit


    result = collections.defaultdict(list)

    for X in range_list:
        np_result = np.array(np.random.multivariate_normal(opt, cov, 500))
        x_res = np.array(list(map(lambda params: naprosyn_adbudg_util(X, params), np_result)))
        result['salesforce'].append(X)
        result['mean'].append(np.mean(x_res))
        result['median'].append(np.median(x_res))
        result['stddev'].append(np.std(x_res))
    
    df = pd.DataFrame(result)
    for i in range(1, 6):
        df[f'U{i}'] = df.apply(lambda x: utility_calc(x.to_dict(), i), axis=1)

    print(df)
