import pandas as pd 
import numpy as np
import pprint
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar, minimize, LinearConstraint, Bounds


p = pprint.PrettyPrinter(depth=6)

cons_df = pd.read_excel('dataset.xlsx', index_col=0, sheet_name='Sheet1')  
m_df = pd.read_excel('dataset.xlsx', index_col=0, sheet_name='Sheet2')  

print(m_df)

proxy_for_infinity = 10
SALESFORCE_COST_PER_PERSON = 0.057


X = [0, 0.5, 1.0, 1.5, proxy_for_infinity]

def ad_budg_eq(x, _min, _max, c, d):
    return _min + (_max - _min) * (x**c) / (d + x**c)

def question_1():
    result = {}
    for col in cons_df:
        col_values = cons_df[col].to_list()
        popt, pcov = curve_fit(ad_budg_eq, X, col_values,\
            p0=[col_values[0], col_values[-1], 1, 1]
        )
        result[col] = popt

    # p.pprint(result)
    return result

max_range = 10_000_000

def question_2():
    q_res = question_1()

    result = {}

    for i, col in enumerate(cons_df):
        _min, _max, c, d = q_res[col]

        def scaler_fn(x):
            x = x[0]
            # print(x, x == np.inf)
            relative_sf = x/(m_df[col]['Current/Original Salesforce'])
            response = _min + (_max - _min) * (relative_sf**c) / (d + relative_sf**c)

            new_profit = response * m_df[col]['Current/Original Revenue'] -\
                (x*SALESFORCE_COST_PER_PERSON)

            return -new_profit

        # res = minimize_scalar(scaler_fn, bounds=(0, np.inf), method='bounded')
        # print('\n', res)

        # return
        print(i)
        lower_bound = 0
        # n_drugs = 8
        # total_salesforce_size = 700
        # x0 = np.ones(n_drugs)*total_salesforce_size/n_drugs
        # sum_constraint_object = LinearConstraint(np.ones(1,n_drugs), lower_bound, total_salesforce_size)
        bounds_object = Bounds(lower_bound, np.inf)
        res = minimize(scaler_fn, 100, method='trust-constr',\
            bounds=bounds_object, options={'verbose': 1})

        print(res.x[0])
        result = {col: res.x[0]}
        # return

        # result = [scaler_fn(i) for i in range(0, 1_000)]
        # plt.plot(result)
        # plt.show()

        # return


