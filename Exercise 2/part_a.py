import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import collections
from multiprocessing import Pool
from functools import partial
from collections import namedtuple, Counter
from utils import timer

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
cutoff = 2.5

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

    
    def is_number(self, attr):
        try:
            return float(self.product_info.get(attr))
        except Exception:
            return False
    
    def is_number_compatible(self, attr):
        if self.is_number(attr):
            return int(self.product_info.get(attr, float('-inf'))) > cutoff
        else:
            return True

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


mugs_data = pd.read_excel('mugs-preference-parameters-full.xlsx', index_col='Cust')

class ProductOptimizer():
    def __init__(self):
        self.get_data()

    def get_data(self):
        df = mugs_data.copy(deep=True)
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

    
    # @timer
    def compute_probability(self):
        for i in range(3):
            self.df[f'exp U{i+1}'] = np.exp(C*self.df[f'U{i+1}'])
        
        self.df[f'Total Exp'] = self.df.iloc[:, -3:].sum(axis=1)

        for i in range(3):
            self.df[f'Prob{i+1}'] = self.df[f'exp U{i+1}'] / self.df[f'Total Exp']
        

    # @timer
    def compute_average_probability(self):
        result = []
        for i in range(3):
            result.append(self.df[f'Prob{i+1}'].mean())
        return result
    
    def get_sorted_importance_parameters_for_customer(self, customer):
        rows = customer.loc[self.importance_features].sort_values(ascending=False)
        rows_dict = rows.to_dict()
        return sorted(rows_dict, key=lambda k: (rows_dict[k], k), reverse=True)

    def cutoffs_with_eba_per_customer(self, customer, products, indx = None):
        sorted_imp = self.get_sorted_importance_parameters_for_customer(customer)

        remaining_products = products

        for imp in sorted_imp:
            attr = self.imp_mapper.get(imp)

            columns = [col for col in self.df.columns if col.startswith(f'p{attr}')]
            filtered_cols = set([pref for pref in columns if customer[pref] > 2.5])

            _products = [
                product for product in remaining_products
                if f'p{attr}{product.get_param(attr)}' in filtered_cols
            ]

            if len(_products) == 1:
                return 1, _products[0]
        
            if not _products:
                random_index = int(random.random() * len(remaining_products))
                return len(remaining_products), remaining_products[random_index]

            remaining_products = _products
        
        random_index = int(random.random() * len(remaining_products))
        return len(remaining_products), (remaining_products[random_index] if indx is None else\
            remaining_products[random_index].index)
    
    def iterate_for_randomness(self, customer, products):
        result = []
        iterations = 100
        
        # func = partial(self.cutoffs_with_eba_per_customer, customer, products)
        # with Pool(5) as p:
        #     result=p.map(
        #         func,
        #         range(iterations)
        #     )

        for _ in range(iterations):
            get_len, product = self.cutoffs_with_eba_per_customer(customer, products)                
            result.append(product.index)
            if get_len == 1:
                break
        
        distributions = Counter(result)
        distributions = {f'P:{k}': v/iterations for k, v in distributions.items()}

        return distributions
        
    # @timer
    def cutoffs_with_eba(self, products):
        self.eba_df = pd.DataFrame(columns = [f'P:{i+1}' for i in range(3)])
        for _, customer in self.df.iterrows():
            distributions = self.iterate_for_randomness(customer, products)
            self.eba_df = self.eba_df.append(distributions, ignore_index=True)

        self.eba_df.fillna(0, inplace=True)
        
        return self.eba_df.mean().values


def compute_expected_profit(market_shares, products):
    return market_shares[-1]*(float(products[-1].price) - products[-1].cost)

# @timer
def optimizer(products):
    obj = ProductOptimizer()

    for product in products:
        obj.utility_of_product(product)

    obj.compute_probability()
    market_shares = obj.compute_average_probability()

    # print(obj.df)

    # for share, product in zip(market_shares, products):
    #     print(share, product.cost, share*product.cost)

    return {
        'expected_profit': compute_expected_profit(market_shares, products),
        'market_share': market_shares[-1]
    }



def multi_process_handle(products, row):
    _, record = row

    prod = Product(record.values, 2, True)
    task = optimizer(products + [prod])
    
    return task



# @timer
def fast_get_all_product_shares():
    products = load_products()
    products.pop()

    products_df = load_products_from_csv()
    record_results = []
    p = Pool(5)
    
    func = partial(multi_process_handle, products)
    record_results=p.map(
        func,
        products_df.iterrows()
        )
    
    products_df['Expected Profit / customer'] = [r.get('expected_profit') for r in record_results]
    products_df['Market Shares'] = [r.get('market_share') for r in record_results]
    
    # print(products_df.head())
    return products_df


def get_all_product_shares():
    products = load_products()
    products.pop()

    products_df = load_products_from_csv()
    record_results = []

    for _, record in products_df.iterrows():
        print('Iteration: ', _)
        prod = Product(record.values, 2, True)
        task = optimizer(products + [prod])

        record_results.append(task)
    
    products_df['Expected Profit / customer'] = [r.get('expected_profit') for r in record_results]
    products_df['Market Shares'] = [r.get('market_share') for r in record_results]
    
    # print(products_df.head())
    return products_df

def main_1():
    products = load_products()
    print(optimizer(products))

def scatter_text(x, y, text_column, data):
    # Create the scatter plot
    p1 = sns.scatterplot(x, y, data=data, size = 8, legend=False, hue='Cn')
    # Add text besides each point
    for line, _ in data.iterrows():
         p1.text(data[x][line], data[y][line]+0.03, 
                 str(line), va='top', ha='center',
                 size='small', color='gray')

    return p1

def main_2(fast=False):
    if fast:
        r1 = fast_get_all_product_shares()

        scatter_text(x="Market Shares", y="Expected Profit / customer",\
            text_column="index", data=r1)
        plt.show()
    else:
        r1 = get_all_product_shares()
        print('done with this')
        print(r1)

        scatter_text(x="Market Shares", y="Expected Profit / customer",\
            text_column="index", data=r1)
        plt.show()


def optimizer_by_elim(products, obj):
    market_shares = obj.cutoffs_with_eba(products)

    return {
        'expected_profit': compute_expected_profit(market_shares, products),
        'market_share': market_shares[-1]
    }

# @timer
def multi_process_handle_elim(products, obj, row):
    _, record = row
    print('Iteration', _)

    prod = Product(record.values, 2, True)
    task = optimizer_by_elim(products + [prod], obj)
    
    return task

def get_all_product_shares_using_elim():
    products = load_products()
    products.pop()

    obj = ProductOptimizer()

    products_df = load_products_from_csv()
    record_results = []
    
    func = partial(multi_process_handle_elim, products, obj)
    
    with Pool(4) as p:
        record_results=p.map(
            func,
            products_df.iterrows()
        )


    # p = Pool(5)
    # func = partial(multi_process_handle_elim, products)
    # 
    # record_results=p.map(
    #     func,
    #     products_df.iterrows()
    # )

    # for _, record in products_df.iterrows():
    #     print('Iteration: ', _)
    #     prod = Product(record.values, 2, True)
    #     task = optimizer_by_elim(products + [prod])

    #     record_results.append(task)
    
    products_df['Expected Profit / customer'] = [r.get('expected_profit') for r in record_results]
    products_df['Market Shares'] = [r.get('market_share') for r in record_results]
    
    print(products_df.head())
    return products_df


def main_3():
    r1 = get_all_product_shares_using_elim()
    print(len(r1))
    scatter_text(x="Market Shares", y="Expected Profit / customer",\
            text_column="index", data=r1)
    plt.show()
    # products = load_products()
    # print(optimizer_by_elim(products))




