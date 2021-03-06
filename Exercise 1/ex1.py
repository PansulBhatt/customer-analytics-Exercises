from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from utils import read_from_file, generate_a_list, compute_coeff, continuous_predict_val,\
        discrete_bass_model, continuous_bass_model, predict_val

def ex_1_a_solution(df):
    """
    Function to estimate the "M", "p" and "q" from the adoption series dataset
    using the linear regression approach
    """
    # Instantiate linear regression model
    lm = LinearRegression()

    # Set the independent and dependent variables
    y = df['N(t)']
    X = df[['A(t)', 'A(t) Squared']]

    # Train the model and fetch the coefficients
    model = lm.fit(X, y)
    a = model.intercept_
    b, c = model.coef_

    # Compute p, q and M, for bass model
    p, q, M = compute_coeff(a, b, c)

    # Perform predictions
    pred_14 = lm.predict([X.iloc[len(X)-1].values])[0]
    print('Prediction for N(14): ', pred_14)
    print(f"p: {p}, q: {q}, M: {M}")
    print('Prediction for N(30): ', predict_val(lm, X.iloc[-1]['A(t)'], pred_14))


def ex_1_b_solution(df, curve_fn=discrete_bass_model, x_param='A(t)'):
    """
    Function to estimate the "p" and "q" from the adoption series dataset
    using the non linear regression approach while keeping M = 100.

    This function can also be used as a baseline to fix the popt values
    for your models
    """
    # Set the independent and dependent variables
    X = df[x_param]
    y = df['N(t)']

    # Compute the coefficients in the non-linear function
    popt, pcov = curve_fit(curve_fn, X, y)
    print('Coefficients: ', popt)
    return popt

def ex_1_c_solution(df):
    """
    Function to predict values from the adoption series dataset
    using the non linear regression approach while holding M = 100.
    """
    # Set the independent and dependent variables
    X = df['A(t)']
    y = df['N(t)']

    # Fetch the coefficients
    p, q = ex_1_b_solution(df)

    # Setup the discrete bass model function with M = 100
    func = lambda x: discrete_bass_model(x, p, q)

    # Fetch predictions
    y_30 = predict_val(None, X.iloc[-1], y.iloc[-1], func)
    print('N(30): ', y_30)

def ex_1_d_solution(df):
    """
    Function to predict values from the adoption series dataset
    using the non linear regression approach while holding M = 100
    while using the continuous bass model.
    """
    # Set the independent and dependent variables
    X = df['t']
    y = df['N(t)']

    p_initial = [0.001, 0.1]

    # Setup the continuous bass model function with M = 100
    popt, pcov = curve_fit(continuous_bass_model, X, y, p0=p_initial)
    print('Coefficients: ', popt)

    p, q = popt
    n_30 = continuous_predict_val(p, q, 30) 
    print('N(30): ', n_30)



def ex_solution():
    """
    Solution to exercise 1
    """
    df = read_from_file('adoptionseries2_with_noise.xlsx')
    generate_a_list(df)

    print('\nExercise 1.1')
    ex_1_a_solution(df)
    print('\nExercise 1.2')
    ex_1_b_solution(df)
    print('\nExercise 1.3 ')
    ex_1_c_solution(df)
    print('\nExercise 1.4')
    ex_1_d_solution(df)