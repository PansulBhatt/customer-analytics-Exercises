import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn import preprocessing
from constants import IMPORTANCE_MAP, PRODUCT_MAP_LIST, PRODUCTS, C, P_LIST, DEMOGRAPHIC
from colorama import Fore, Back, Style


# mugs_df = pd.read_excel('data/mugs-preference-parameters-full.xlsx')
# mugs_df.rename(columns=lambda x: x.strip(), inplace=True)
# mugs_df_columns = mugs_df.columns

# def calculate_mugs_parameters():
#     for imp, p_cols in IMPORTANCE_MAP.items():
#         for p_col in p_cols:
#             mugs_df[f'I*{p_col}'] = mugs_df[imp] * mugs_df[p_col]

#     for i, product in enumerate(PRODUCTS):
#         mugs_df[f'U{i+1}'] = mugs_df.filter([f'I*p{PRODUCT_MAP_LIST[i]}{product[i]}'\
#             for i in range(len(product))]).sum(axis=1)
#         mugs_df[f'exp{i+1}'] = np.exp(mugs_df[f'U{i+1}']*C)

#     mugs_df['Total exp'] = mugs_df.filter([f'exp{i+1}' for i in range(3)]).sum(axis=1)
#     for i in range(3):
#         mugs_df[f'P{i+1}'] = mugs_df[f'exp{i+1}']/mugs_df['Total exp']
    
#     for i, product in enumerate(PRODUCTS):
#         for imp in P_LIST:
#             mugs_df[f'P{i+1}*{imp}'] = mugs_df[f'P{i+1}'] * mugs_df[imp]

#     return mugs_df[[f'P{i+1}' for i in range(3)]]

# prob_df = calculate_mugs_parameters()

### From the excel files

mugs_df = pd.read_excel('data/mugs-analysis-full-incl-demographics.xlsx',
                sheet_name='mugs-full', skiprows=29)
mugs_df.rename(columns=lambda x: x.strip(), inplace=True)
prob_df = mugs_df[['P1', 'P2', 'P3']].copy(deep=True)

cols = [col for col in mugs_df if col not in ['P1', 'P2', 'P3'] and not col.startswith('Unnamed')]
mugs_df = mugs_df[cols]

cluster_analysis_df = pd.read_excel('data/mugs-analysis-full-incl-demographics.xlsx',
                sheet_name='for-cluster-analysis', index='Cust')
cluster_analysis_df.rename(columns=lambda x: x.strip(), inplace=True)

columns = cluster_analysis_df.columns
for i, product in enumerate(PRODUCTS):
    for col in columns:
        cluster_analysis_df[f'P{i+1}*{col}'] = prob_df[f'P{i+1}'] * cluster_analysis_df[col]


sum_of_descriptors = cluster_analysis_df.sum().to_dict()
mean_for_descriptors = cluster_analysis_df.mean(axis=0).to_dict()


def log_lift_fn(x):
    return np.log10(x / mean_for_descriptors[x.name])

def part_1():
    """
    computes the weighted average of all the columns in the
    "for-cluster-analysis" worksheet, weighted by the probabilities given in column BF of the
    "mugs-full" worksheet.
    """
    # Calculate probability sum
    prob_sum_dict = prob_df.sum().to_dict()

    data = {}
    # Iterate through the items to get the segementation mean
    for index in range(3):
        i = index + 1
        prefix = f'P{i}'
        probability_sum = prob_sum_dict[prefix]
        data[f'Seg.{i}.mean'] = {characteristic: \
            (sum_of_descriptors[f'{prefix}*{characteristic}'] / probability_sum)  \
            for characteristic in P_LIST + DEMOGRAPHIC
        }

    print(json.dumps(data, indent=2))
    # Store results in affinity segment
    affinity_segmentation_profile = pd.DataFrame.from_dict(data, orient='index')
    print(affinity_segmentation_profile)

    log_fits_df = affinity_segmentation_profile.copy(deep=True)

    log_fits_df = log_fits_df.apply(log_lift_fn)
    log_fits_df.to_csv('log_lift_part_a.csv')

    print(log_fits_df)


def perform_k_means_clustering(n_clusters, values):
    model = KMeans(n_clusters=n_clusters, n_init=50, max_iter=100, random_state=410014)
    model.fit(values)
    return model

def plot_elbow_curve(cluster_range, avg_list):
    plt.plot(cluster_range, avg_list, 'bx-') 
    plt.xlabel('Values of K') 
    plt.ylabel('Average within cluster')
    plt.show() 


def profiling_set_average(profiling_df):
    return profiling_df.groupby(['segment']).mean()


def k_means_cluster_analysis():
    """
    Performs k-means cluster analysis
    """
    analysis_df = cluster_analysis_df.copy(deep=True)
    values = analysis_df[P_LIST].values

    # Try plotting the curve to evaluate the best model
    cluster_range = range(1, 11)

    avg_list = [perform_k_means_clustering(n, values).inertia_ / len(P_LIST)\
        for n in cluster_range]

    plot_elbow_curve(cluster_range, avg_list)

    # After evaluation, we see that the best model is the one that takes 6 clusters.
    best_kmeans = perform_k_means_clustering(6, values)
    best_kmeans.fit(values)

    # Create segmentation and profiling dataframe
    analysis_df['segment'] = [i+1 for i in best_kmeans.labels_]

    segment_df = pd.DataFrame.from_records(best_kmeans.cluster_centers_)
    segment_df.columns = P_LIST
    segment_df.index += 1

    profiling_df = profiling_set_average(analysis_df[DEMOGRAPHIC + ['segment']])

    segment_df.to_csv('segment_characteristics.csv')
    profiling_df.to_csv('profiling_k_means.csv')

    return segment_df, profiling_df

def log_lift_computation(segment_df, profiling_df):
    """
    This function is used to compute the log lift
    """
    log_lift_df = pd.concat([segment_df, profiling_df], axis=1, join='inner')
    log_lift_df = log_lift_df.apply(log_lift_fn)
    return log_lift_df

def color_code(val):
    """
    This function is used for showing the log lifts with highlights
    """
    color = Fore.BLACK
    if val > 1:
        color = Fore.GREEN
    elif val < 0:
        color = Fore.RED
    return color + f'{val}' + Style.RESET_ALL

def part_2():
    segment_df, profiling_df = k_means_cluster_analysis()
    log_lift_df = log_lift_computation(segment_df, profiling_df)
    log_lift_df.to_csv('log_lift_k_means.csv')
    # Feel free to se this to display the columns in the terminal
    # In jupyter, you would not have to set this.

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    log_lift_df = log_lift_df.applymap(color_code)
    print(log_lift_df)
    

def main():
    print('Part 1:')
    part_1()
    print('\n\nPart 2:')
    part_2()


# Optional part

def standardize():
    names = cluster_analysis_df.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(cluster_analysis_df)
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    return scaled_df


def optional_3():
    X = standardize()
    # X is an np.array containing the data
    kmeansModel = KMeans(n_clusters=6, n_init=50, max_iter=100)
    kmeansModel.fit(X)
    labels = kmeansModel.labels_
    pca_2 = PCA(2)
    # Standardize the X matrix to make PCA operate on correlations instead of covariances
    X_std = (X - np.mean(X)) / np.std(X)
    plot_columns = pca_2.fit_transform(X_std)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
    plt.colorbar()
    plt.show()


def optional_4():
    """
    We are trying to use cosine similarity to understand which customers align
    best with other customers. Then we will compare those customers with the ones
    which have a preference for our brand and compare if they have a higher probability
    to go with our brand.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import collections
    _df = mugs_df.copy(deep=True)

    similarity = cosine_similarity(_df)

    # Priority customers are those who prefer our brand most
    priority_customers = prob_df.sort_values(['P3'], ascending=False)
    priority_customers.reset_index(inplace=True)

    brand_pref = collections.defaultdict(int)

    for i in range(36):
        baseline_customer_index = int(priority_customers.iloc[i]['index'])
        
        cosine_scores = [None] * len(mugs_df)
        for index, val in enumerate(similarity[baseline_customer_index-1]):
            cosine_scores[index] = (val, index)

        sorted_scores_with_customer_id = list(sorted(cosine_scores, reverse=True))

        for score, index in sorted_scores_with_customer_id:
            if score < 0.9:
                break
            key = prob_df.loc[index].idxmax(axis=1)
            brand_pref[key] += 1
    
    print('brand preference\n', brand_pref, '\n', '*'*40, '\n')

    print(prob_df.describe())
    # check how many people prefer brand C
    prob_df['Max Prob'] = prob_df.idxmax(axis=1)
    print(collections.Counter(prob_df['Max Prob']))

def optional():
    optional_3()
    optional_4()