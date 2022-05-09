from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



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



# Adapted from https://www.kaggle.com/dgawlik/house-prices-eda#Categorical-data
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)


def boxplots_categorical_columns(df, categorical_columns, dependant_variable):
    """
    Takes df, a list of categorical columns, a dependant variable as str
    Returns group boxplots of correlations between categorical varibles and dependant variable
    """

    f = pd.melt(df, id_vars=[dependant_variable], value_vars=categorical_columns)
    g = sns.FacetGrid(f, col='variable',  col_wrap=2, sharex=False, sharey=False, height=10)
    g = g.map(boxplot, 'value', dependant_variable)
    return g


def heatmap_numeric_w_dependent_variable(df, dependent_variable):
    '''
    Takes df, a dependant variable as str
    Returns a heatmap of independent variables' correlations with dependent variable
    '''
    plt.figure(figsize=(8, 10))
    g = sns.heatmap(df.corr()[[dependent_variable]].sort_values(by=dependent_variable),
                    annot=True,
                    cmap='coolwarm',
                    vmin=-1,
                    vmax=1)
    return g


def plot_feature_importance(importances):
    # importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Plot the feature importancies of the forest
    num_to_plot = 10
    feature_indices = [ind+1 for ind in indices[:num_to_plot]]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(num_to_plot):
        print("%d. %s %f " % (f + 1,
                features["f"+str(feature_indices[f])],
                importances[indices[f]]))

    plt.title(u"Feature Importance")
    bars = plt.bar(range(num_to_plot), importances[indices[:num_to_plot]],
           color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]), align="center")
    ticks = plt.xticks(range(num_to_plot), feature_indices)
    plt.xlim([-1, num_to_plot])
    plt.legend(bars, [u''.join(features["f"+str(i)]) for i in feature_indices])


def freq_words(vocab, terms = 30):
    fdist = FreqDist(vocab)
    words_df = pd.DataFrame({'word':list(fdist.keys()),
             'count':list(fdist.values())})
    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms)
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()
