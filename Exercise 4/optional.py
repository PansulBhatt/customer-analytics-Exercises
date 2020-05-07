import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from factor_analyzer import FactorAnalyzer
from mandatory import plot_util, COLORS
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from gensim import matutils, models, corpora
import scipy.sparse
import pyLDAvis
import pyLDAvis.gensim as gensimvis

lemmatizer = WordNetLemmatizer()

asins = pd.read_csv('data/amazon-face-treatment-reviews/asins.csv')
reviews = pd.read_csv('data/amazon-face-treatment-reviews/reviews.csv')
reviews.columns = reviews.columns.str.replace(' ', '')

words = reviews.groupby(['asin'])['fullreview'].apply(lambda x: ' '.join(x)).reset_index()

bow = pd.read_csv('data/amazon-face-treatment-reviews/bag-of-words-representation.csv')
bow.columns = [lemmatizer.lemmatize(col) for col in bow.columns]
# add the duplicate columns and then remove the duplicates
bow = bow.sum(axis=1, level=0)
bow = bow.loc[:,~bow.columns.duplicated()]

df = bow.copy(deep=True)
df['asin'] = reviews['asin']

df = df.groupby(['asin'])[bow.columns].apply(lambda x : x.astype(int).sum())
df = df.reset_index()
asin_col = df.pop('asin')
df.pop('Unnamed: 0')

def factor_analyzer(n_factors):
    fa = FactorAnalyzer(n_factors=n_factors, rotation=None)
    fa_fit_out = fa.fit(df)
    fa_communalities = fa_fit_out.get_communalities()
    fa_gof = sum(fa_communalities)
    fa_scores = fa_fit_out.transform(df)
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
    k = pd.DataFrame(points, columns=['X', 'Y'])
    plt.scatter(k['X'], k['Y'])
    plt.show()

def plot_using_ldaviz(topics=10):
    texts = [[text] for text in bow.columns]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lda = models.LdaModel(corpus_tfidf, num_topics=topics, id2word=dictionary)
    vis_data = gensimvis.prepare(lda, corpus, dictionary)
    pyLDAvis.show(vis_data)

def plot_using_lda():
    # Find the best model using cross validation
    print('running cv')
    search_params = {'n_components': [2, 5, 10], 'learning_decay': [.3, .5]}
    lda = LatentDirichletAllocation(random_state=0)
    model = GridSearchCV(lda, param_grid=search_params)
    model.fit(df.values)

    best_lda_model = model.best_estimator_
    print(model.best_params_)

    points = best_lda_model.transform(df.values)
    k = pd.DataFrame(points, columns=['X', 'Y'])
    k['dominant_topic'] = np.argmax(k.values, axis=1)
    k['asin'] = asin_col
    plt.scatter(k['X'], k['Y'])
    # Normal plot without directions
    plot_util(points, k['dominant_topic'].values.tolist(), 0.1)
    plt.show()

def main():
    plot_using_factor_analysis(2)
    plot_using_lda()
    plot_using_ldaviz(2)

