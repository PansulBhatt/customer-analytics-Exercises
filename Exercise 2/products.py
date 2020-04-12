from collections import namedtuple
from constants import PRICING, PRODUCTS, CONTAINMENT_MAP, PRODUCT_MAP_LIST, C, CUTOFF

MyProduct = namedtuple('MyProduct', PRODUCT_MAP_LIST)

class Product():
    """
    Class to contain the product details
    """
    def __init__(self, data, index, primary=False):
        self.product_info = MyProduct._make(data)._asdict()
        self.price = self.product_info.get('Pr')
        self.index = index+1
        self.primary = primary
        self.compute_cost()
    
    def compute_cost(self):
        cost = 0
        for k, prices in PRICING.items():
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
            return int(self.product_info.get(attr, float('-inf'))) > CUTOFF
        else:
            return True

    def __repr__(self):
        return str(self.product_info)