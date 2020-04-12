import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from multiprocessing import Pool
from functools import partial
from collections import namedtuple, Counter
from utils import timer
from products import Product
from compute import ProductOptimizer
from constants import PRICING, PRODUCTS, CONTAINMENT_MAP, PRODUCT_MAP_LIST, C

# Making certain values global so I don't have to keep refetching values
MyProduct = namedtuple('MyProduct', PRODUCT_MAP_LIST)

def load_products():
    """
    Function to load all the products and push them to the product object.
    The Product class contains most of the information about the product
    and we can fetch all such values directly from the appropriate values
    """
    result = []
    for index, row in enumerate(PRODUCTS):
        obj = Product(row, index, index == len(PRODUCTS) - 1)
        result.append(obj)
    return result

def load_products_from_csv():
    """
    Function to load the data from the CSV file to our products_df dataframe.
    We have to make sure that the values are passed correctly and contain
    the appropriate data for our analysis.
    """
    df = pd.read_csv('products.csv', names = ['index'] + PRODUCT_MAP_LIST,\
        header=None, index_col='index')

    for p in PRODUCT_MAP_LIST[:3]:
        # This should extract the required data as numbers & also adjust for floats.
        df[p] = df[p].str.extract('(\d*\.\d+|\d+)')

    # We need our Prices to be updated with a 0 if there is only 1 digit
    df['Pr'] = df['Pr'].apply(lambda x: f'0{x}' if len(x) == 1 else x)
    # Update containment to their respective maps
    df['Cn'] = df['Cn'].apply(lambda x: CONTAINMENT_MAP[x.strip()])

    df['Cl'] = df['Cl'].apply(lambda x: x.strip()[0])
    # Brand should be C
    df['Br'] = 'C'
    
    return df


def compute_expected_profit(market_shares: list, products: list):
    """
    Computes expected profit by checking the last values (as we append the comparison product
    at the end)
    """
    return market_shares[-1]*(float(products[-1].price) - products[-1].cost)

# @timer
def optimizer(products: list):
    """
    This function performs all the necessary operations for us.
    Steps:
    1. Start by setting up the product optimizer class which contains
    most of the data
    2. Iterate through the products to calculate their utility
    3. Compute probability market share
    """
    # Step 1
    obj = ProductOptimizer()

    # Step 2
    for product in products:
        obj.utility_of_product(product)

    # Step 3
    obj.compute_probability()
    market_shares = obj.compute_average_probability()

    # print(obj.df['pBrC'].describe())
    # print(obj.df['pBrA'].describe())

    # print(obj.df)

    # for share, product in zip(market_shares, products):
    #     print(share, product.cost, share*product.cost)

    return {
        'expected_profit': compute_expected_profit(market_shares, products),
        'market_share': market_shares[-1],
        'cost': products[-1].cost
    }



def multi_process_handle(products, row):
    """
    Multiprocess handler, this function takes in the required value
    and computes based on the requirements. Each process would
    essentially call this function for processing
    """
    # Unpack the element data
    _, record = row

    # Initialize the product and execute the optimizer function
    prod = Product(record.values, 2, True)
    task = optimizer(products + [prod])
    
    return task


# @timer
def fast_get_all_product_shares():
    """
    Multiprocessing way of handling analysis.
    """
    # Load required data
    products = load_products()
    products.pop()

    products_df = load_products_from_csv()
    record_results = []

    # Setup our pool of processes
    p = Pool(5)
    
    # Setup function to take in extra parameters
    func = partial(multi_process_handle, products)

    # Execute pool map of products_df
    record_results=p.map(
        func,
        products_df.iterrows()
        )
    
    # Compute expected profit
    products_df['Expected Profit / customer'] = [r.get('expected_profit') for r in record_results]
    products_df['Market Shares'] = [r.get('market_share') for r in record_results]
    products_df['Cost'] = [r.get('cost') for r in record_results]
    
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
    products_df['Cost'] = [r.get('cost') for r in record_results]
    
    # print(products_df.head())
    return products_df

def optimizer_by_elim(products, obj):
    """
    Optimize function which executes eliminations
    """
    market_shares = obj.cutoffs_with_eba(products)

    return {
        'expected_profit': compute_expected_profit(market_shares, products),
        'market_share': market_shares[-1]
    }

# @timer
def multi_process_handle_elim(products, obj, row):
    """
    Multiprocess handler, this function takes in the required value
    and computes based on the requirements. Each process would
    essentially call this function for processing. This functions
    task is to optimize based on elimination
    """
    _, record = row
    print('Iteration', _)

    prod = Product(record.values, 2, True)
    task = optimizer_by_elim(products + [prod], obj)
    
    return task

def get_all_product_shares_using_elim():
    """
    Computes the product shares by using EBA.
    """
    # Load all the required products
    products = load_products()
    # Pop the last value which is to replaced with the other product combination
    products.pop()

    # Instantiate product optimizer
    obj = ProductOptimizer()

    products_df = load_products_from_csv()
    record_results = []

    # Setup a partial function
    # This has been done so that we can send extra arguments in our multi-processed operations    
    func = partial(multi_process_handle_elim, products, obj)
    
    # Execute pool
    with Pool(4) as p:
        record_results=p.map(
            func,
            products_df.iterrows()
        )
    
    # Compute the required data from the recorded results
    products_df['Expected Profit / customer'] = [r.get('expected_profit') for r in record_results]
    products_df['Market Shares'] = [r.get('market_share') for r in record_results]
    products_df['Cost'] = [r.get('cost') for r in record_results]
    
    print(products_df.head())
    return products_df


def scatter_text(x, y, text_column, data):
    # Create the scatter plot
    print('comes here')
    p1 = sns.scatterplot(x, y, data=data, size = 8, legend=False, hue='Cn')
    # Add text besides each point
    for line, _ in data.iterrows():
         p1.text(data[x][line], data[y][line]+0.03, 
                 str(line), va='top', ha='center',
                 size='small', color='gray')

    return p1


def main_1():
    """
    Using the compensatory rule with logit adjustment:
    Compute and report our candidate's share, price, margin and expected profit
    per customer under the "proposed market scenario"
    """
    products = load_products()
    print(optimizer(products))
    print(products[-1].price, products[-1].cost)


def main_2(fast=False):
    """
    Discrete Optimization
    """
    if fast:
        r1 = fast_get_all_product_shares()
        print(r1)
        r1.to_csv('main_b.csv')
        scatter_text(x="Market Shares", y="Expected Profit / customer",\
            text_column="index", data=r1)
        plt.show()
    else:
        r1 = get_all_product_shares()
        print(r1)
        r1.to_csv('main_b.csv')
        scatter_text(x="Market Shares", y="Expected Profit / customer",\
            text_column="index", data=r1)
        plt.show()

def main_3():
    """
    elimination-by-aspects (EBA)
    """
    r1 = get_all_product_shares_using_elim()
    r1.to_csv('main_d.csv')
    print(len(r1))
    scatter_text(x="Market Shares", y="Expected Profit / customer",\
            text_column="index", data=r1)
    plt.show()




