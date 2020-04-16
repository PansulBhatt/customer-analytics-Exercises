IMPORTANCE_MAP = {
    'IPr': ['pPr30', 'pPr10', 'pPr05'],
    'Iin': ['pIn0.5', 'pIn1', 'pIn3'],
    'ICp': ['pCp12', 'pCp20', 'pCp32'],
    'ICl': ['pClD', 'pClF', 'pClE'],
    'Icn': ['pCnSl', 'pCnSp', 'pCnLk'],
    'IBr': ['pBrA', 'pBrB', 'pBrC']
}

PRODUCTS = [
    ['30', '3', '20', 'E', 'Lk', 'A'],
    ['10', '1', '20', 'F', 'Sp', 'B'],
    ['30', '3', '20', 'E', 'Lk', 'C']
]

PRODUCT_MAP_LIST = ['Pr', 'In', 'Cp', 'Cl', 'Cn', 'Br']

C=0.0139

P_LIST = ['IPr', 'Iin', 'ICp', 'ICl', 'Icn', 'IBr', 'I*pPr30', 'I*pPr10', 'I*pPr05',\
    'I*pIn0.5', 'I*pIn1', 'I*pIn3', 'I*pCp12', 'I*pCp20', 'I*pCp32', 'I*pClD',\
    'I*pClF', 'I*pClE', 'I*pCnSl', 'I*pCnSp', 'I*pCnLk', 'I*pBrA', 'I*pBrB', 'I*pBrC']

DEMOGRAPHIC = ['income', 'age', 'sports', 'gradschl']