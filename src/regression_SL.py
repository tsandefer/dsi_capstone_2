import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# LETS DO SOME REGRESSION STUFF!

def split_for_holdout(X, y):
    # Make an initial training/holdout split so we can test how generalizable our model is later
    # Using an 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def post_split_eda(m_train, m_holdout, p_train, p_holdout):
    # Look @ correlation Matrix
x_corr = full_df_Math[math_predictors].corr()
plt.matshow(x_corr)
x_corr

    sns.pairplot(p_apprx_cont_df)


    # Residual plot
    y_vals.plot(kind='scatter', y='residuals', x='fitted_vals')

    # QQ PLOT
    fig, ax = plt.subplots(figsize=(12, 5))
    fig = sm.qqplot(stu_resid, line='45', fit=True, ax=ax)

    # PLOT THE CONFUSION Matrix
    cfm = confusion_matrix(predicted_classes, y_test.values)
    sns.heatmap(cfm, annot=True)
    plt.xlabel('Predicted classes')
    plt.ylabel('Actual classes')

    # DATA VIZ
    num_attributes.hist(figsize=(10,10))
    train_data.describe()


def get_xy():
    x_cols = ['primary_contributor_IQ',
              'n_tate_contributors',
              'comment_cnt',
              'song_tate_cnt',
              'n_unreviewed_tates',
              'chars_in_tate',
              'chars_in_referent',
              'ref_word_cnt',
              'tate_word_cnt']
    y_col = 'votes_per_1000views'

    X = df[x_cols]
    y = df[y_col]
    X = sm.add_constant(X)
    return X, y

model = sm.OLS(Y,X)
results = model.fit()

print(results.summary())

if __name__ == '__main__':

    X, y = get_xy()
