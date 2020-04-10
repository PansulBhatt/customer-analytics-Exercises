import asyncio
import numpy as np
import pandas as pd
import collections
from collections import namedtuple
"""
Price: $30, $10, $5
Time Insulated: 0.5 hrs, 1 hrs, 3 hrs
Capacity: 12 oz, 20 oz, 32 oz
Cleanability: Difficult (7 min), Fair (5 min), Easy (2 min)
Containment: Slosh resistant, Spill resistant, Leak resistant
Brand: A, B, C
"""
# [
#     {
#         'Price': 30,
#         'Insulation': 3,
#         'Capacity': 20,
#         'Cleanability': 'ClE',
#         Leak Resistant, Brand A
#     }
# ]

pricing = {
    'In': {'0.5': 0.5, '1': 1, '3': 3},
    'Cp': {'12': 1, '20': 2.6, '32': 2.8},
    'Cl': {'D': 1, 'F': 2.2, 'E': 3},
    'Cn': {'Sl': 0.5, 'Sp': 0.8, 'Lk': 1}
}

products = [
    ['30', '3', '20', 'E', 'Lk', 'A'],
    ['10', '1', '20', 'F', 'Sp', 'B'],
    ['30', '1', '20', 'E', 'Lk', 'C']
]

containment_map = {
    'Slosh resistant': 'Sl',
    'Spill resistant': 'Sp',
    'Leak resistant': 'Lk'
}

product_map_list = ['Pr', 'In', 'Cp', 'Cl', 'Cn', 'Br']
MyProduct = namedtuple('MyProduct', product_map_list)

C=0.0139

class Product():
    def __init__(self, data, index, primary=False):
        self.product_info = MyProduct._make(data)._asdict()
        self.price = self.product_info.get('Pr')
        self.index = index+1
        self.primary = primary
        self.compute_cost()
    
    def compute_cost(self):
        cost = 0
        for k, prices in pricing.items():
            attr = self.get_param(k)
            cost += prices[attr]
        
        self.cost = cost

    def get_param(self, attr):
        return self.product_info.get(attr, None)

    def __repr__(self):
        return str(self.product_info)

def load_products():
    result = []
    for index, row in enumerate(products):
        obj = Product(row, index, index == len(products) - 1)
        result.append(obj)
    return result

def load_products_from_csv():
    df = pd.read_csv('products.csv', names = ['index'] + product_map_list,\
        header=None, index_col='index')

    for p in product_map_list[:3]:
        df[p] = df[p].str.extract('(\d*\.\d+|\d+)')

    df['Pr'] = df['Pr'].apply(lambda x: f'0{x}' if len(x) == 1 else x)
    df['Cn'] = df['Cn'].apply(lambda x: containment_map[x.strip()])
    df['Cl'] = df['Cl'].apply(lambda x: x.strip()[0])
    df['Br'] = 'C'
    
    return df

class ProductOptimizer():
    def __init__(self):
        self.get_data()

    def get_data(self):
        df = pd.read_excel('mugs-preference-parameters-full.xlsx', index_col='Cust')
        df.rename(columns=lambda x: x.strip(), inplace=True)
        self.importance_features = [col for col in df if col.startswith('I')]
        self.product_features = [col for col in df if not col.startswith('I')]
        self.imp_mapper = dict(zip(self.importance_features, product_map_list))
        self.df = df

    def set_product_utility_header(self, index, series):
        self.df[f'U{index}'] = series

    def get_importance_parameters(self):
        return self.importance_features

    def utility_of_product(self, product):
        series = []
        for _, customer in self.df.iterrows():
            val = self.utility_of_product_for_customer(customer, product)
            series.append(val)
        
        self.set_product_utility_header(product.index, series)

    def utility_of_product_for_customer(self, customer, product):
        product_utility = 0
        for importance_feature in self.importance_features:
            importance_factor = customer[importance_feature]
            pr_key = self.imp_mapper[importance_feature]
            col_key = f'p{pr_key}{product.get_param(pr_key)}'
            product_utility += customer[col_key]*importance_factor

        return product_utility

    def compute_probability(self):
        for i in range(3):
            self.df[f'exp U{i+1}'] = np.exp(C*self.df[f'U{i+1}'])
        
        self.df[f'Total Exp'] = self.df.iloc[:, -3:].sum(axis=1)

        for i in range(3):
            self.df[f'Prob{i+1}'] = self.df[f'exp U{i+1}'] / self.df[f'Total Exp']
        

    def compute_average_probability(self):
        result = []
        for i in range(3):
            result.append(self.df[f'Prob{i+1}'].mean())
        return result


def compute_expected_profit(market_shares, products):
    return market_shares[-1]*products[-1].cost

def optimizer(products):
    obj = ProductOptimizer()

    for product in products:
        # print(product.cost)
        obj.utility_of_product(product)
    obj.compute_probability()
    market_shares = obj.compute_average_probability()

    for share, product in zip(market_shares, products):
        print(share, product.cost, share*product.cost)

    return compute_expected_profit(market_shares, products)


async def fast_optimizer(products):
    obj = ProductOptimizer()

    for product in products:
        # print(product.cost)
        obj.utility_of_product(product)
    obj.compute_probability()
    market_shares = obj.compute_average_probability()


    return compute_expected_profit(market_shares, products)


def main_1():
    products = load_products()
    optimizer(products)

import time

async def get_all_product_shares():
    products = load_products()
    products.pop()

    products_df = load_products_from_csv()
    record_results = []
    for _, record in products_df.iterrows():
        print('Iteration: ', _)
        prod = Product(record.values, 2, True)
        task = asyncio.create_task(fast_optimizer(products + [prod]))
        await task
        record_results.append(task.result)
    
    products_df['Expected Profit / customer'] = record_results
    
    print(products_df.head())
    
def main_2():
    time1 = time.time()
    asyncio.run(get_all_product_shares())
    time2 = time.time()
    print(time2 - time1)
