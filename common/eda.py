from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def view_columns_w_many_nans(df: pd.DataFrame, missing_limit: int=.9) -> List[str]:
    """
    Checks which columns have over specified percentage of missing values
    Returns columns as a list
    """
    percent_missing = df.isnull().mean()
    series = percent_missing[percent_missing > missing_limit]
    columns = series.index.to_list()
    return columns


def drop_columns_w_many_nans(df: pd.DataFrame, missing_limit: int=.9) -> pd.DataFrame:
    """
    Drops the columns whose missing value is bigger than missing percentage
    """
    cols_with_missing = view_columns_w_many_nans(df, missing_percent=missing_limit)
    list_of_cols = cols_with_missing.index.to_list()
    return df.drop(columns=list_of_cols)


# Adapted from https://www.kaggle.com/dgawlik/house-prices-eda#Categorical-data
# Reference: https://seaborn.pydata.org/tutorial/axis_grids.html
def histograms_numeric_columns(df: pd.DataFrame, numerical_columns: List[str]):
    """
    Takes df, numerical columns as list
    Returns group histagrams
    """
    f = pd.melt(df, value_vars=numerical_columns)
    g = sns.FacetGrid(f, col='variable',  col_wrap=4, sharex=False, sharey=False)
    g = g.map(sns.distplot, 'value')
    return g


####################### to refactor

def categorical_to_ordinal_transformer(categories):
    """
    Returns a function that will map categories to ordinal values based on the
    order of the list of `categories` given. Ex.
    If categories is ['A', 'B', 'C'] then the transformer will map
    'A' -> 0, 'B' -> 1, 'C' -> 2.
    """
    return lambda categorical_value: categories.index(categorical_value)



def transform_categorical_to_numercial(df, categorical_numerical_mapping):
    '''
    Transforms categorical columns to numerical columns
    Takes a df, a dictionary
    Returns df
    '''
    transformers = {k: categorical_to_ordinal_transformer(v)
                    for k, v in categorical_numerical_mapping.items()}
    new_df = df.copy()
    for col, transformer in transformers.items():
        new_df[col] = new_df[col].map(transformer).astype('int64')
    return new_df


def dummify_categorical_columns(df):
    '''
    Dummifies all categorical columns
    Takes df
    Returns df
    '''
    categorical_columns = df.select_dtypes(include="object").columns
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)


def error_metrics(y_true, y_preds, n, k):
    '''
    Takes y_true, y_preds,
    n: the number of observations.
    k: the number of independent variables, excluding the constant.
    Returns 6 error metrics
    '''
    def r2_adj(y_true, y_preds, n, k):
        rss = np.sum((y_true - y_preds)**2)
        null_model = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - rss/null_model
        r2_adj = 1 - ((1-r2)*(n-1))/(n-k-1)
        return r2_adj

    print('Mean Square Error: ', mean_squared_error(y_true, y_preds))
    print('Root Mean Square Error: ', np.sqrt(mean_squared_error(y_true, y_preds)))
    print('Mean absolute error: ', mean_absolute_error(y_true, y_preds))
    print('Median absolute error: ', median_absolute_error(y_true, y_preds))
    print('R^2 score:', r2_score(y_true, y_preds))
    print('Adjusted R^2 score:', r2_adj(y_true, y_preds, n, k))


# Adapted from https://www.kaggle.com/dgawlik/house-prices-eda#Categorical-data
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)


# def boxplots_categorical_columns(df, categorical_columns, dependant_variable):
#     """
#     Takes df, a list of categorical columns, a dependant variable as str
#     Returns group boxplots of correlations between categorical varibles and dependant variable
#     """
#
#     f = pd.melt(df, id_vars=[dependant_variable], value_vars=categorical_columns)
#     g = sns.FacetGrid(f, col='variable',  col_wrap=2, sharex=False, sharey=False, height=10)
#     g = g.map(boxplot, 'value', dependant_variable)
#     return g


# def heatmap_numeric_w_dependent_variable(df, dependent_variable):
#     '''
#     Takes df, a dependant variable as str
#     Returns a heatmap of independent variables' correlations with dependent variable
#     '''
#     plt.figure(figsize=(8, 10))
#     g = sns.heatmap(df.corr()[[dependent_variable]].sort_values(by=dependent_variable),
#                     annot=True,
#                     cmap='coolwarm',
#                     vmin=-1,
#                     vmax=1)
#     return g


# def plot_feature_importance(importances):
#     # importances = forest.feature_importances_
#     indices = np.argsort(importances)[::-1]
#     # Plot the feature importancies of the forest
#     num_to_plot = 10
#     feature_indices = [ind+1 for ind in indices[:num_to_plot]]
#
#     # Print the feature ranking
#     print("Feature ranking:")
#     for f in range(num_to_plot):
#         print("%d. %s %f " % (f + 1,
#                 features["f"+str(feature_indices[f])],
#                 importances[indices[f]]))
#
#     plt.title(u"Feature Importance")
#     bars = plt.bar(range(num_to_plot), importances[indices[:num_to_plot]],
#            color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]), align="center")
#     ticks = plt.xticks(range(num_to_plot), feature_indices)
#     plt.xlim([-1, num_to_plot])
#     plt.legend(bars, [u''.join(features["f"+str(i)]) for i in feature_indices])


# def freq_words(x, terms = 30):
#   all_words = ' '.join([text for text in x])
#   all_words = all_words.split()
# 
#   fdist = FreqDist(all_words)
#   words_df = pd.DataFrame({'word':list(fdist.keys()),
#              'count':list(fdist.values())})
#   # selecting top 20 most frequent words
#   d = words_df.nlargest(columns="count", n = terms)
#   plt.figure(figsize=(20,5))
#   ax = sns.barplot(data=d, x= "word", y = "count")
#   ax.set(ylabel = 'Count')
#   plt.show()
