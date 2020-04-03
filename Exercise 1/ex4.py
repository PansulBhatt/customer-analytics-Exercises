import matplotlib.pyplot as plt 
from constants import PQ_PAIRS
from utils import read_from_file, generate_a_list, predict_val, compute_coeff, continuous_predict_val,\
    discrete_bass_model

def discrete_computation(df, p, q):
    X = df['A(t)']
    y = df['N(t)']

    func = lambda x: discrete_bass_model(x, p, q)
    y_hat = predict_val(None, 0, 100*q, func, range(0, 30), list_return=True)
    return y_hat

def continuous_computation(df, p, q):
    X = df['t']
    y = df['N(t)']

    y_hat = continuous_predict_val(p, q, 30, list_return=True)
    return y_hat


def ex_solution():
    df = read_from_file('adoptionseries2_with_noise.xlsx')
    generate_a_list(df)

    fig, axs = plt.subplots(nrows=len(PQ_PAIRS)//2, ncols=2)
    fig.tight_layout(pad=3.0)

    print('\nExercise 4')

    for index, (p, q) in enumerate(PQ_PAIRS):
        d = discrete_computation(df, p, q)
        c = continuous_computation(df, p, q)

        y = 1 if index >= 4 else 0

        axs[index%4, y].plot(d)
        axs[index%4, y].plot(c)
        axs[index%4, y].set_title(f"With P: {p} and Q: {q}")
        

    plt.show()