import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.manifold import MDS
from factor_analyzer import FactorAnalyzer


cars_od = pd.read_csv("data/cars.dissimilarity.csv", sep=",", index_col=0)
cars_ar = pd.read_csv("data/cars.ar.csv", sep=",", index_col=0)

COLORS = ['salmon', 'palegreen', 'deepskyblue', 'crimson', 'orange', 'lawngreen', 'teal']

def scatter_text(x, y, data, text_columns, offset):
    # Create the scatter plot
    p1 = sns.scatterplot(x, y, data=data)
    # Add text on top of each point
    for line, _ in data.iterrows():
         p1.text(data[x][line], data[y][line]+offset, 
                 text_columns[line], va='top', ha='center',
                 size='small', color='gray')

    return p1

def mds_util(k, metric):
    mds = MDS(n_components=k, metric=metric,\
        max_iter=1000, eps=1e-9, dissimilarity="precomputed",\
        n_jobs=1, random_state=3)

    mds_fit_out = mds.fit(cars_od)

    return  {
        'stress': mds_fit_out.stress_,
        'embedding': mds_fit_out.embedding_
    }

def question_util_a(range_list, metric):
    stress_results = [mds_util(index, metric).get('stress') for index in range_list]
    plt.plot(range_list, stress_results)
    plt.show()

def plot_util(points, text_columns=cars_od.columns, offset=0.3):
    df = pd.DataFrame(points, columns=['X', 'Y'])
    scatter_text('X', 'Y', df, text_columns, offset)
    xmin, xmax, ymin, ymax = plt.axis('square')
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))

def question_util_b(metric, offset=0.3):
    points = mds_util(2, metric).get('embedding')
    plot_util(points, cars_od.columns, offset)
    plt.show()

def factor_analyzer(n_factors):
    fa = FactorAnalyzer(n_factors=n_factors, rotation=None)
    fa_fit_out = fa.fit(cars_ar)
    fa_communalities = fa_fit_out.get_communalities()
    fa_gof = sum(fa_communalities)
    fa_scores = fa_fit_out.transform(cars_ar)
    fa_factor_loadings = fa_fit_out.loadings_
    return {
        'fa_gof': fa_gof,
        'fa_communalities': fa_communalities,
        'fa_scores': fa_scores,
        'fa_factor_loadings': fa_factor_loadings
    }

def plot_using_factor_analysis(metric):
    data = factor_analyzer(metric)
    points = data['fa_scores']
    r2s = data['fa_communalities']
    coeffs = data['fa_factor_loadings']

    # Normal plot without directions
    plot_util(points, cars_ar.index, 0.1)
    limit = 2.5
    plt.xlim((-limit, limit))
    plt.ylim((-limit, limit))
    plt.show()

    # with directions
    plot_util(points, cars_ar.index, 0.1)
    i = 0
    for r2, coeff, col in zip(r2s, coeffs, cars_ar.columns):
        x, y = coeff
        color = COLORS[i]
        i = (i+1) % len(COLORS)
        arrow_end_x = 1.5*r2*x / np.sqrt(x**2 + y**2)
        arrow_end_y = 1.5*r2*y / np.sqrt(x**2 + y**2)
        plt.arrow(0, 0, arrow_end_x, arrow_end_y, color=color)
        plt.text(arrow_end_x, arrow_end_y, 
                 col, va='top', ha='center',
                 size='small', color=color)

    plt.xlim((-limit, limit))
    plt.ylim((-limit, limit))
    plt.show()

## DRIVERS 
def problem_1():
    question_util_a(list(range(1, 10)), True)
    question_util_b(True)

def problem_2():
    question_util_a(list(range(1, 6)), False)
    question_util_b(False, 0.05)

def problem_3():
    # Factor analysis
    gofs = [factor_analyzer(i)['fa_gof'] for i in range(1, 11)]
    plt.plot(list(range(1, 11)), gofs)
    plt.show()
    plot_using_factor_analysis(2)

