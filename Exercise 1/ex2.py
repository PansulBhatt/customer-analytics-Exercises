import matplotlib.pyplot as plt 
from utils import read_from_file, generate_a_list, compute_coeff, continuous_predict_val,\
        discrete_bass_model, continuous_bass_model, predict_val

from constants import PQ_PAIRS

def ex_solution():
    """
    Solution to exercise 2:
    The Effect of Changes in Relative Values of p, q
    """
    df = read_from_file('adoptionseries2_with_noise.xlsx')
    generate_a_list(df)

    M = 100

    fig, axs = plt.subplots(4, 2)
    fig.tight_layout(pad=3.0)

    print('\nExercise 2')

    for index, (p, q) in enumerate(PQ_PAIRS):
        func = lambda x: discrete_bass_model(x, p, q)

        y_hat = predict_val(None, 0, M*p, func, for_range=range(1, 30), list_return=True)

        y = 1 if index >= 4 else 0
        axs[index%4, y].plot(y_hat)
        axs[index%4, y].set_ylim([0, 12])
        axs[index%4, y].set_title(f"P: {p}, Q: {q}")

    plt.show()