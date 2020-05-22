import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

"""
Direct Mail Prospect Scoring Exercise
"""

train_df = pd.read_excel('prospectscoringhw.xlsx', skiprows=2, nrows=200)
holdout_df = pd.read_excel('prospectscoringhw.xlsx', skiprows=206, nrows=300)

y_train = train_df.pop('y')
y_train = y_train.values
X_train = train_df.values

y_test = holdout_df['y']
y_test = y_test.values
X_test = holdout_df.loc[:, ~holdout_df.columns.isin(['y'])]

clf = LogisticRegression(penalty='none', max_iter=1000).fit(X_train, y_train)


# 1: The score equation for t from the logistic regression.
print(clf.intercept_)
print(dict(zip(train_df.columns, clf.coef_[0])))

# 2: Get probabilities and lift

holdout_df['probabilities'] = None
holdout_df['probabilities'] = clf.predict_proba(X_test)[:, 1]

holdout_df['r'] = holdout_df['probabilities']

mean = holdout_df['r'].mean()
holdout_df['lift'] = holdout_df.apply(lambda x: x['r']/mean, axis=1)


# 3:  Sort all the persons in the holdout list in decreasing order of lift as is done in slide 30.
holdout_sorted_df = holdout_df.sort_values(by=['lift'], ascending=False)

# 4: Plot the curve for marginal response rate vs number of solicitations made
plt.scatter(list(range(len(holdout_sorted_df))), holdout_sorted_df['r'])
plt.show()


# 5. marginal cost rule : pn > solicitation cost / Equity = 178
equity = 30
cost = 12

min_prob = cost/equity

num = sum(holdout_df['r'] > min_prob)
print(num)


# 6. Cummumlative sum : No huge rise due to a less sample and less increase in
# the probability. Feel that the dataset might be more distributed in comparison to the real world
holdout_sorted_df['cummsum'] = holdout_sorted_df['probabilities'].cumsum()
holdout_sorted_df = holdout_sorted_df.sort_values(by=['cummsum'])

plt.scatter(list(range(len(holdout_sorted_df))), holdout_sorted_df['cummsum'])
plt.show()


# 7. use limited supply rule to distribute 40 boxes => 64 people
distribution_df = holdout_sorted_df[holdout_sorted_df['cummsum'] <= 40]
print(distribution_df.shape[0])

# 8. Test accuracy
# Superimpose plots
plt.scatter(list(range(len(holdout_sorted_df))), holdout_sorted_df['y'].cumsum())
plt.scatter(list(range(len(holdout_sorted_df))), holdout_sorted_df['cummsum'])
plt.show()




