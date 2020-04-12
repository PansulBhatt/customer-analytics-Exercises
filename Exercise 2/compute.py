import random
import numpy as np
import pandas as pd
from collections import Counter
from constants import PRICING, PRODUCTS, CONTAINMENT_MAP, PRODUCT_MAP_LIST, C, CUTOFF

mugs_data = pd.read_excel('mugs-preference-parameters-full.xlsx', index_col='Cust')

class ProductOptimizer():
    """
    This class is the main class for this exercise. The idea behind maintaing
    a class is that we can fetch the required parameters more easily.

    # TODO: Find a way to integrate the product class in this using abstraction.
    """
    def __init__(self):
        # On initialization we fetch the required values and push them to a df
        self.get_data()

    def get_data(self):
        """
        Fetch the data from the global variable (mugs_data)
        and update the data as per our requirements
        """
        df = mugs_data.copy(deep=True)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        self.importance_features = [col for col in df if col.startswith('I')]
        self.product_features = [col for col in df if not col.startswith('I')]
        self.imp_mapper = dict(zip(self.importance_features, PRODUCT_MAP_LIST))
        self.df = df

    def set_product_utility_header(self, index, series):
        """
        Function to add a new column to our data (the column here is the utility column)
        """
        self.df[f'U{index}'] = series

    def get_importance_parameters(self):
        return self.importance_features
    
    def utility_of_product(self, product):
        """
        Function to compute the utility of a product.
        To calculate this, we must iterate through all the customers
        and update based on their results.
        """
        series = []
        for _, customer in self.df.iterrows():
            val = self.utility_of_product_for_customer(customer, product)
            series.append(val)
        
        self.set_product_utility_header(product.index, series)

    def utility_of_product_for_customer(self, customer, product):
        """
        Function to compute the utility of a product for 1 customer.
        """
        product_utility = 0
        # Iterate through all the importance features
        for importance_feature in self.importance_features:
            # fetch the value in the importance feature
            importance_factor = customer[importance_feature]
            # Map the key using the importance mapper. This should allow us to get
            # which details we need to extract from the product
            pr_key = self.imp_mapper[importance_feature]
            # Initialize key which we need to check in our df.
            col_key = f'p{pr_key}{product.get_param(pr_key)}'
            # Increment overall product utility
            product_utility += customer[col_key]*importance_factor

        return product_utility

    
    # @timer
    def compute_probability(self):
        """
        Function to compute the probability using the utility columns
        
        P(purchase) = exp(c*up) / (exp(c*up) + exp(c*ui) + ...)
        """
        for i in range(3):
            self.df[f'exp U{i+1}'] = np.exp(C*self.df[f'U{i+1}'])
        
        self.df[f'Total Exp'] = self.df.iloc[:, -3:].sum(axis=1)

        for i in range(3):
            self.df[f'Prob{i+1}'] = self.df[f'exp U{i+1}'] / self.df[f'Total Exp']
        

    def compute_average_probability(self):
        """
        Function to compute average probability across
        """
        result = []
        for i in range(3):
            result.append(self.df[f'Prob{i+1}'].mean())
        
        return result
    
    def get_sorted_importance_parameters_for_customer(self, customer):
        rows = customer.loc[self.importance_features].sort_values(ascending=False)
        rows_dict = rows.to_dict()
        return sorted(rows_dict, key=lambda k: (rows_dict[k], random.random(), k), reverse=True)

    def cutoffs_with_eba_per_customer(self, customer, products, indx = None):
        """
        Compute the cutoff using EBA per customer for all the products

        Params:
        customer: customer object with individual customer details
        products: products list with which we are performing our EBA
        indx: Index

        Return Values:
        Tuple (<length of remaining products>, <product index which was selected>)
        """
        # Get sorted importance features
        sorted_imp = self.get_sorted_importance_parameters_for_customer(customer)

        # Maintain a list of remaining products
        # which can be used for analysis
        remaining_products = products

        # Iterate through all the sorted importance features
        for imp in sorted_imp:
            # Fetch the mapped attribute
            attr = self.imp_mapper.get(imp)

            # Fetch all required columns which start with the attribute
            # TODO: Optimize this up if time permits.
            columns = [col for col in self.df.columns if col.startswith(f'p{attr}')]

            # Maintain a set of filtered columns which are greater than cutoff
            filtered_cols = set([pref for pref in columns if customer[pref] > CUTOFF])

            # Map products with filtered columns
            # and save it to a temporary array. This will allow
            # us to return values while also maintaining the original remaining products array
            # In essense, _products is a subset of remaining_products, per iteration
            _products = [
                product for product in remaining_products
                if f'p{attr}{product.get_param(attr)}' in filtered_cols
            ]

            # If length of products is 1, then simply return (No need to iterate anymore)
            if len(_products) == 1:
                return 1, _products[0]
        
            # If none of the products were selected, then see if we can randomly
            # choose a value from the original remaining products
            if not _products:
                random_index = int(random.random() * len(remaining_products))
                return len(remaining_products), remaining_products[random_index]

            # Update remaining products
            remaining_products = _products
        
        # If we were not able to eliminate the products, then randomly select a value
        random_index = int(random.random() * len(remaining_products))
        return len(remaining_products), (remaining_products[random_index] if indx is None else\
            remaining_products[random_index].index)
    
    def iterate_for_randomness(self, customer, products):
        """
        Iteration to take care of the randomness. This will allow us to understand
        the distributions as well.
        """
        result = []
        iterations = 100
        
        # Iterate a 100 times but break the loop if there was only 1 product which was found.
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
        """
        Compute the cutoffs using EBA
        """
        # Set eba_df which can contain the required information about the data
        self.eba_df = pd.DataFrame(columns = [f'P:{i+1}' for i in range(3)])

        # Iterate through all the customers and append based on their preferences
        for _, customer in self.df.iterrows():
            distributions = self.iterate_for_randomness(customer, products)
            self.eba_df = self.eba_df.append(distributions, ignore_index=True)

        # Remove all NaN with 0 for mean.
        self.eba_df.fillna(0, inplace=True)
        
        return self.eba_df.mean().values