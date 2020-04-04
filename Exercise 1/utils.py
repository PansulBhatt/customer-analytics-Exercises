import xlrd
import numpy as np
import pandas as pd
from scipy.special import expit
from constants import DATA_DIR

def read_from_file(file_name):
    """
    Function to read file excel files
    """
    # Base condition
    if not file_name:
        return

    # Read excel file and return the response
    df = pd.read_excel('{0}{1}'.format(DATA_DIR, file_name))
    return df

def generate_a_list(df):
    """
    Function to update the df. Since df is going to be passed by reference
    we can make the updates directly to df and don't need to return anything else.
    """
    # Initialize the required values
    a_t = [0]

    # Iterate through all the rows
    for index, row in df.iterrows():
        if index == len(df) - 1:
            continue

        val = row['N(t)'] + a_t[-1]
        a_t.append(val)
    
    # Update the DF
    df['A(t)'] = a_t
    df['A(t) Squared'] = df['A(t)'].apply(lambda x: x**2)

def compute_coeff(a, b, c, M=None):
    """
    Function to compute the coefficients from the mathematical formula
    """
    if not M:
        D = b**2 - 4*a*c

        p, q = (D**0.5 - b) / 2, (D**0.5 + b) / 2

        return p, q, -q/c
    else:
        return a/M, c*M, M

def predict_val(lm, a_t: int, n_t: int, predict_fn=None, for_range=range(14, 30), list_return=False):
    """
    Function to perform predictions on N(t) using the A(t) using discrete bass model.
    This function also allows the provision to define the prediction function, in which
    case we can switch the mathematical formula based on our requirements. 

    Arguments:
    lm: Linear Regression model after training (not necessary to provide if you have a predict_fn)
    a_t: A(t) column in the DF
    n_t: N(t) column in the DF
    predict_fn: Prediction function which can perform computations for the next value
    for_range: Range till when you want your predictions
    list_return: Boolean value to notify the function on the return type

    Returns:
    Either a list of predictions (based on the list_return arg) or the last prediction value

    Formula:
    N(t) = a + bA(t) + c[A(t)]^2
    A(t) = N(t-1) + A(t-1)
    """
    if a_t is None or n_t is None:
        return

    n_list = [n_t]

    for i in for_range:
        # A(t) = N(t-1) + A(t-1)
        a_t = n_t + a_t
        # N(t) = a + bA(t) + c[A(t)]^2
        if not lm:
            n_t = predict_fn(a_t)
        else:
            n_t = lm.predict([[a_t, a_t**2]])[0]
        
        n_list.append(n_t)

    return n_t if not list_return else n_list

def continuous_predict_val(p, q, predict_on=14, initial=0, list_return=False):   
    """
    Function to predict using the continuous bass model. This function iterates through the values
    and stores the result in a list. It keeps on using continuous bass model function to make the
    next predictions and moves forward till it hits the limit (defined by predict_on)

    Arguments:
    p: Coefficient p
    q: Coefficent q
    predict_on: Limit for your predictions
    initial: From when you want to start your loop
    list_return: Boolean value to notify the function on the return type

    Returns:
    Either a list of predictions (based on the list_return arg) or the last prediction value

    """ 
    n_t = []
    for i in range(initial+1, initial+predict_on):
        n_t.append(continuous_bass_model(i, p, q))

    return n_t[-1] if not list_return else n_t

def discrete_bass_model(x, p, q):
    """
    N(t) = M*p + (q − p)A(t) − (q/M)*[A(t)]2
    """
    return 100*p + (q-p)*x - (q/100)*(x**2)

def continuous_bass_model(t, p, q):
    """
    A(t) = M* 1 − exp(−(p + q)t)/ 1 + qp exp(−(p + q)t)
    """
    return 100*(1-np.exp(-(p+q)*t))/(1+(q/p)*np.exp(-(p+q)*t)) -\
            100*(1-np.exp(-(p+q)*(t-1)))/(1+(q/p)*np.exp(-(p+q)*(t-1)))