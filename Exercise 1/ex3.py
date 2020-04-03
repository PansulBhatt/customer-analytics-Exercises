import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from utils import read_from_file, generate_a_list, predict_val, compute_coeff

np.random.seed(1729)

def generate_predictions(original_df, add_noise=0):
    lm = LinearRegression()

    df = original_df.copy(deep=False)
    y = df['N(t)']
    X = df[['A(t)', 'A(t) Squared']]

    if add_noise:
        y_noise = add_noise * np.random.normal(size=y.size)
        y += y_noise

    model = lm.fit(X, y)
    a = model.intercept_
    b, c = model.coef_

    p, q, M = compute_coeff(a, b, c)

    return (predict_val(lm, 0, M*p, for_range=range(1, 30), list_return=True))

def create_plot_from_dataset(file_name, sub_plot, randomize=0.1):
    df = read_from_file(file_name)
    generate_a_list(df)

    y_hat = generate_predictions(df)
    sub_plot.plot(y_hat)

    y_hat = generate_predictions(df, randomize)
    sub_plot.plot(y_hat)

    sub_plot.set_title(f"From {file_name} with randomization: {randomize}")



def ex_solution():
    """
    Solution to exercise 3
    """
    fig, axs = plt.subplots(2, 2)


    print('\nExercise 3')    
    create_plot_from_dataset('adoptionseries1.xlsx', axs[0, 0])
    create_plot_from_dataset('adoptionseries2_with_noise.xlsx', axs[0, 1])

    create_plot_from_dataset('adoptionseries1.xlsx', axs[1, 0], 0.2)
    create_plot_from_dataset('adoptionseries2_with_noise.xlsx', axs[1, 1], 0.2)

    plt.show()

"""
Randomizing the data does have an impact on how the data is predicted. If we randomize
the data and factor the values accordingly then the value of M is affected greatly. If
the predictions don't take care of the M, the curve peak would be greatly impacted. If 
the data has a lot of noise it would add problems during the prediction and thus the
curve would not be approximated properly.
"""