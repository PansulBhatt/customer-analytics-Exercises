import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from . import cons_df, m_df
from .mandatory import question_1
import random

np.random.seed(4100142)
SALESFORCE_COST_PER_PERSON = 0.057

def range_util(start, stop, count):
    step = (stop - start) / float(count)
    return [start + i * step for i in range(count)]

def calc_util(x, vol):
    return x - 0.5*2*vol

def optional_part_1():
    range_list = range_util(100, 441, 11)
    original_sf = m_df['Naprosyn']['Current/Original Salesforce']
    original_revenue = m_df['Naprosyn']['Current/Original Revenue']
    relative_sfs = [i/original_sf
        for i in range_list
    ]
    mn, *_, mx = cons_df['Naprosyn'] # calculates min and max
    print(mn, mx)

    result = question_1()
    mn, mx, c, d = result['Naprosyn']

    def naprosyn_adbudg_util(x):
        response = mn + (mx - mn) * (x**c) / (d + x**c)
        new_revenue = response*original_revenue
        new_profit = new_revenue*0.7 -  x*SALESFORCE_COST_PER_PERSON
        return new_profit

    results = (list(map(naprosyn_adbudg_util, relative_sfs)))
    print(results)

    theta_hat, varcov_theta_hat = curve_fit(calc_util, range_list, results, p0=[0.5099])
    
    np_result = np.array(np.random.multivariate_normal(theta_hat, varcov_theta_hat, 500))
    print(np_result.std())
    
