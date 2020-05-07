import json
from functools import wraps
from time import time
from colorama import Fore, Style

def listToTuple(function):
    def wrapper(*args):
        args = [json.dumps(x) if type(x) != str else x for x in args]
        result = function(*args)
        result = json.dumps(result) if type(result) == list else result
        return result
    return wrapper

def timer(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(Fore.GREEN + f'Function: {f.__name__}, Elapsed time: {(end-start)/1000:.2f}')
        print(Style.RESET_ALL)
        return result
    return wrapper
