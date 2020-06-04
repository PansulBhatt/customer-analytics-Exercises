import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta, t

volumes_df = pd.read_excel('data/volumes.dataset.2.xlsx', index_col='cust')
clicks_df = pd.read_excel('data/clicks.dataset.2.xlsx', index_col='ad')

NUMBER_OF_DRAWS_NEEDED = 100_000

result_df_1 = pd.DataFrame()
result_df_2 = pd.DataFrame()


def generate_draws(a, b):
    return beta.rvs(a, b, size=NUMBER_OF_DRAWS_NEEDED, random_state=0)

def generate_t_distribution_draws(degrees=NUMBER_OF_DRAWS_NEEDED+1):
    return t.rvs(df=degrees, size=NUMBER_OF_DRAWS_NEEDED, random_state=0)

def compute_averages(df, prefix='Ad'):
    df['max'] = df.max(axis=1)
    averages = {}
    for col in clicks_df:
        df[f'is_max_{col}'] = (df['max']==df[f'{prefix}:{col}']).astype(int)
        averages[col] = df[f'is_max_{col}'].mean()
    
    print("\033[034m", averages, "\033[0m", "\n\n")
    return averages

def question_1():
    global result_df_1
    generated_res = {}
    for col in clicks_df:
        beta_s = clicks_df[col]['clicks'] + 1
        beta_p = clicks_df[col]['exposures'] - clicks_df[col]['clicks'] + 1
        generated_res[f'Ad:{col}'] = generate_draws(beta_s, beta_p)
    
    result_df_1 = pd.DataFrame.from_dict(generated_res)
    compute_averages(result_df_1)

def question_2():
    generated_res = {}

    for idx, group in volumes_df.groupby('ad'):
        generated_res[idx] = {
            'minimum': group['volume'].min(),
            'maximum': group['volume'].max(),
            'mean': group['volume'].mean(),
            'median': group['volume'].median(),
            'variance': group['volume'].var(),
            'std': group['volume'].std(),
            'std.err': group['volume'].std() / np.sqrt(len(group['volume'])),
            'bayes.posterior.df': len(group['volume']) + 1
        }

    for ad in generated_res:
        mean = generated_res[ad]['mean']
        std_err = generated_res[ad]['std.err']
        bayes_num = generated_res[ad]['bayes.posterior.df']
        result_df_2[f'Ad:{ad}'] = generate_t_distribution_draws(bayes_num)
        result_df_2[f'Ad:{ad}'] = result_df_2[f'Ad:{ad}']*std_err + mean

    compute_averages(result_df_2)
        

def question_3():
    result_df_3 = pd.DataFrame()
    result_df_3[[f'ctr_draws:{i}' for i in range(1, 6)]] = result_df_1[[f'Ad:{i}' for i in range(1, 6)]]
    result_df_3[[f'mu_draws:{i}' for i in range(1, 6)]] = result_df_2[[f'Ad:{i}' for i in range(1, 6)]]

    for i in range(1, 6):
        result_df_3[f'ctr*mu:{i}'] = result_df_3[f'mu_draws:{i}'] * result_df_3[f'ctr_draws:{i}']

    result_df_3['max'] = result_df_3[[f'ctr*mu:{i}' for i in range(1, 6)]].max(axis=1)
    main_df = result_df_3[[f'ctr*mu:{i}' for i in range(1, 6)]].copy(deep=True)
    compute_averages(main_df, 'ctr*mu')



if __name__ == '__main__':
    print('Answer 1:')
    question_1()
    print('Answer 2:')
    question_2()
    print('Answer 3:')
    question_3()