import pandas as pd 
import numpy as np
import pprint

p = pprint.PrettyPrinter(depth=6)  

cons_df = pd.read_excel('dataset.xlsx', index_col=0, sheet_name='Sheet1')  
m_df = pd.read_excel('dataset.xlsx', index_col=0, sheet_name='Sheet2')