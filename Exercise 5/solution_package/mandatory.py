import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar, minimize, LinearConstraint, Bounds
from . import cons_df, m_df, p

proxy_for_infinity = 10
SALESFORCE_COST_PER_PERSON = 0.057

X = [0, 0.5, 1.0, 1.5, proxy_for_infinity]


def ad_budg_eq(x, _min, _max, c, d):
    return _min + (_max - _min) * (x**c) / (d + x**c)

def question_1(cov = False):
    result = {}
    for col in cons_df:
        col_values = cons_df[col].to_list()
        popt, pcov = curve_fit(ad_budg_eq, X, col_values,\
            p0=[col_values[0], col_values[-1], 1, 1]
        )
        result[col] = popt if not cov else {'popt': popt, 'pcov': pcov}

    return result


q_res = question_1()

def question_2():
    result = {}

    for i, col in enumerate(cons_df):
        _min, _max, c, d = q_res[col]

        def scaler_fn(x):
            x = x[0]
            relative_sf = x/(m_df[col]['Current/Original Salesforce'])
            response = _min + (_max - _min) * (relative_sf**c) / (d + relative_sf**c)
            new_revenue = response*m_df[col]['Current/Original Revenue']
            new_profit = new_revenue * m_df[col]['Profit Margin'] -\
                (x*SALESFORCE_COST_PER_PERSON)

            return -new_profit

        lower_bound = 0
        bounds_object = Bounds(lower_bound, np.inf)
        res = minimize(scaler_fn, 80, method='trust-constr',\
            bounds=bounds_object, options={'verbose': 1})

        result[col] = {
            'max-profit # salespersons: 2': res.x[0],
            'objective function value: 2': -res.fun
        }

    df = pd.DataFrame(result)
    print('\n\n\n\n', df.head())
    print('\n'*3)
    p.pprint(result)

    return df



def negative_profit(x):
    new_profit = 0
    for i, col in enumerate(cons_df):
        _min, _max, c, d = q_res[col]
        el = x[i]
        relative_sf = el/(m_df[col]['Current/Original Salesforce'])
        response = _min + (_max - _min) * (relative_sf**c) / (d + relative_sf**c)
        new_revenue = response*m_df[col]['Current/Original Revenue']
        new_profit += new_revenue * m_df[col]['Profit Margin'] -\
            (el*SALESFORCE_COST_PER_PERSON)
        
    return -new_profit

def question_3():
    n_drugs = 8
    total_salesforce_size = 700
    lower_bound = 0
    x0 = np.ones(n_drugs)*total_salesforce_size/n_drugs
    sum_constraint_object = LinearConstraint(np.ones((1, n_drugs)), lower_bound, total_salesforce_size)
    bounds_object = Bounds(lower_bound, np.inf)
    optimizer_output = minimize(negative_profit, x0, method='SLSQP',\
            bounds=bounds_object,
            constraints=sum_constraint_object
            )

    print('\n'*4, optimizer_output, '\n'*4)
    df = pd.DataFrame(optimizer_output.x, index=cons_df.columns, columns=['optimal_val']).T
    df = pd.concat([df, question_2()]).T

    df['Relative decrease in SF'] = \
        (df['max-profit # salespersons: 2'] - df['optimal_val']) / df['max-profit # salespersons: 2']
    df['Relative decrease in SF %'] = df['Relative decrease in SF'] * 100
    print(df)
