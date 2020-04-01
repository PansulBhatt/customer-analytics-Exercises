import xlrd
import pandas as pd
from constants import DATA_DIR

def read_from_file(file_name):
    """
    Function to read file excel files
    """
    if not file_name:
        return

    df = pd.read_excel('{0}{1}'.format(DATA_DIR, file_name))
    print(df.head())

