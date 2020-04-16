import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans
from collections import defaultdict
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

cluster_analysis_df = pd.read_excel('data/mugs-analysis-full-incl-demographics.xlsx',
                sheet_name='for-cluster-analysis', index='Cust')
cluster_analysis_df.rename(columns=lambda x: x.strip(), inplace=True)

sum_of_descriptors = cluster_analysis_df.sum().to_dict()
mean_for_descriptors = cluster_analysis_df.mean(axis=0).to_dict()

columns = cluster_analysis_df.columns
for i, product in enumerate(PRODUCTS):
    for col in columns:
        cluster_analysis_df[f'P{i+1}*{col}'] = prob_df[f'P{i+1}'] * cluster_analysis_df[col]


def log_lift_fn(x):
    return np.log(x / mean_for_descriptors[x.name])

def part_1():
    # print(mugs_df.columns)
    # Ipr: -.427

    print(cluster_analysis_df.head())
    # print(mean_for_descriptors)
    print()
    # print(sum_of_descriptors)
    prob_sum_dict = prob_df.sum().to_dict()
    print()
    print()
    print(prob_sum_dict)

    data = {}
    for index in range(3):
        i = index + 1
        prefix = f'P{i}'
        probability_sum = prob_sum_dict[prefix]
        data[f'Seg.{i}.mean'] = {characteristic: \
            (sum_of_descriptors[f'{prefix}*{characteristic}'] / probability_sum)  \
            for characteristic in P_LIST
        }

    affinity_segmentation_profile = pd.DataFrame.from_dict(data, orient='index')
    print(affinity_segmentation_profile)

    log_fits_df = affinity_segmentation_profile.copy(deep=True)

    log_fits_df = log_fits_df.apply(log_lift_fn)

    print(log_fits_df)
    # analysis = defaultdict(dict)

    # for key, val in mugs_df.mean(axis=0).items():
    #     if not key.startswith('P'):
    #         continue
    #     try:
    #         prob, param = key.split('*')
    #         analysis[param][prob] = val
    #     except:
    #         pass

    # print(analysis)


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
    analysis_df = cluster_analysis_df.copy(deep=True)
    values = analysis_df[P_LIST].values

    cluster_range = range(1, 11)
    segmentation_basis_set = analysis_df[P_LIST].columns

    # avg_list = [perform_k_means_clustering(n, values).inertia_ / len(P_LIST)\
    #     for n in cluster_range]

    # plot_elbow_curve(cluster_range, avg_list)

    best_kmeans = perform_k_means_clustering(6, values)
    best_kmeans.fit(values)

    analysis_df['segment'] = best_kmeans.labels_

    segment_df = pd.DataFrame.from_records(best_kmeans.cluster_centers_)
    segment_df.columns = P_LIST
    # print(segment_df)

    profiling_df = profiling_set_average(analysis_df[DEMOGRAPHIC + ['segment']])
    # print(profiling_df)

    return segment_df, profiling_df

def log_lift_computation(segment_df, profiling_df):
    log_lift_df = pd.concat([segment_df, profiling_df], axis=1, join='inner')
    log_lift_df = log_lift_df.apply(log_lift_fn)
    print(log_lift_df)
    return log_lift_df

def color_code(val):
    if val > 1:
        color = Fore.GREEN
    elif val < 0:
        color = Fore.RED
    return color + str('{0:.2%}'.format(val)) + Style.RESET_ALL

def part_2():
    segment_df, profiling_df = k_means_cluster_analysis()
    log_lift_df = log_lift_computation(segment_df, profiling_df)
    log_lift_df = log_lift_df.style.applymap(color_code)
    log_lift_df.render()
    

def main():
    # part_1()
    part_2()
    return 