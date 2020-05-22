import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict
from scipy.optimize import curve_fit, minimize, Bounds, NonlinearConstraint
from bisect import bisect_left


pp = pprint.PrettyPrinter(indent=4)

ltv_df = pd.read_excel('data/hw-kw-ltv-conv.rate-data.xlsx')
keyword_list = ltv_df['keyword'].to_list()

clicksdata = []
for keyword in keyword_list:
    clicksdata.append(pd.read_csv(f'data/clicksdata.kw.all/clicksdata.{keyword}.csv', index_col=0))


def number_of_clicks(x, a, b):
    return a*(1-np.exp(-b*x))

def get_closest_value_for_b2(df, val):
    idx = bisect_left(df['n.clicks'], val)
    return np.log(2) / df.iloc[idx]['bid.value']

def exercise_a():
    results = {}
    for index, df in enumerate(clicksdata):
        mx = max(df['n.clicks'])
        p_initial = [mx, get_closest_value_for_b2(df, mx/2)]

        popt, pcov = curve_fit(number_of_clicks,
                                df['bid.value'], df['n.clicks'],
                                p0=p_initial,
                                bounds=(0, [float('inf'), float('inf')]))
        results[keyword_list[index]] = popt    
    
    res_df = pd.DataFrame(results).T
    res_df.columns = ['alpha', 'beta']
    print(res_df)

    return results, res_df

df_results, res_a_df = exercise_a()

def profit_overall(x, a, b, ltv, conversion):
    return number_of_clicks(x, a, b) * (ltv*conversion - x)

def total_expenditure(x, a, b):
    return x*a*(1-np.exp(-b*x))

def exercise_b():
    result_dict = defaultdict(dict)
    for i, row in ltv_df.iterrows():
        a, b = df_results[row['keyword']]
        ltv, conversion = row['ltv'], row['conv.rate']

        def compute_fn(x):
            return -profit_overall(x, a, b, ltv, conversion)

        results = []

        for x0 in range(0, 100, 10):
            res = minimize(compute_fn, x0, method='trust-constr')
            results.append(res)
        
        best_idx = np.argmax(list(map(lambda x: -x.fun[0], results)))
        result_dict[row['keyword']] = {
            'best_bid': results[best_idx].x[0],
            'optimal_profit': -results[best_idx].fun[0],
            'expenditure': total_expenditure(results[best_idx].x[0], a, b)
        }

    pp.pprint(result_dict)
    ans_b_df = pd.DataFrame(result_dict).T
    print("\033[35m", ans_b_df.head(), "\033[0m") 
    return ans_b_df


def constraint_func(x):
    const_sum = 0
    for i, keyword in enumerate(keyword_list):
        a, b = df_results[keyword]
        const_sum += x[i]*number_of_clicks(x[i], a, b)
    return const_sum

def exercise_c():
    def max_profit(x_list):
        profit_so_far = 0
        
        for i, x in enumerate(x_list):
            row = ltv_df.iloc[i]
            a, b = df_results[row['keyword']]
            ltv, conversion = row['ltv'], row['conv.rate']
            profit_so_far += profit_overall(x, a, b, ltv, conversion)
        
        return -profit_so_far

    cons = NonlinearConstraint(constraint_func, 0, 3000)
    res = minimize(max_profit, [0]*4, method='trust-constr',
                    bounds=Bounds(0, np.inf),
                    constraints=cons
    )

    expenditures = []
    for i, row in ltv_df.iterrows():
        a, b = df_results[row['keyword']]
        expenditures.append(res.x[i] * number_of_clicks(res.x[i], a, b))

    res_df_c = pd.DataFrame({'bid': res.x, 'expenditure': expenditures},
                            index = keyword_list)

    print("\033[34m", "Profit: ", -res.fun, "\033[0m") 
    print(res_df_c)
    return res_df_c


def exercise_d(ans_b_df):
    temp_df = res_a_df.copy()
    temp_df['ltv'] = ltv_df['ltv'].values
    temp_df['best_bid'] = ans_b_df['best_bid'].values
    temp_df = temp_df.sort_values(by=['ltv'])
    print("\033[32m", temp_df, "\033[0m") 


def exercise_e(ans_b_df, ans_c_df):
    print((ans_b_df['expenditure'] - ans_c_df['expenditure']) / ans_b_df['expenditure'])
    pass

if __name__ == '__main__':
    # exercise_a()
    print('\n'*3)
    print('Answer B')
    ans_b_df = exercise_b()
    print('\n'*3)
    print('Answer C')
    ans_c_df = exercise_c()
    print('\n'*3)
    print('Answer D')
    exercise_d(ans_b_df)
    print('\n'*3)
    print('Answer E')
    exercise_e(ans_b_df, ans_c_df)