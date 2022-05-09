from typing import List, Dict, Union, Optional
import numpy as np
import pandas as pd
from functools import reduce
from itertools import zip_longest
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer
np.set_printoptions(precision=3)
pd.set_option('display.width', 100)


def dummify_categorical_columns(df, categorical_columns=[]):
    "Dummifies all categorical columns"
    if not categorical_columns:
        categorical_columns = df.select_dtypes(include="object").columns
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)


def map_categorical_to_numercial(categories):
    """
    Returns a function that will map categories to ordinal values based on the
    order of the list of `categories` given. Ex.
    If categories is ['A', 'B', 'C'] then the transformer will map
    'A' -> 0, 'B' -> 1, 'C' -> 2.
    """
    return lambda categorical_value: categories.index(categorical_value)


def transform_categorical_to_numercial(df, categorical_numerical_mapping):
    "Transforms categorical columns to numerical columns"
    transformers = {k: convert_categorical_to_numeric(v)
                    for k, v in categorical_numerical_mapping.items()}
    new_df = df.copy()
    for col, transformer in transformers.items():
        new_df[col] = new_df[col].map(transformer).astype('int64')
    return new_df


def merge_dataframes_by_columns(dfs: List[pd.DataFrame], key_columns=Union[str, List[str]], join_as: str=['left', 'right', 'inner', 'outer']
                               )-> pd.DataFrame:
    """
    Merge multiple dataframe by common column shared. Doesn't work if column name differs across multiple dataframe.
    sourced from: https://stackoverflow.com/questions/23668427/pandas-three-way-joining-multiple-dataframes-on-columns
    """
    merged_df = reduce(df_left, df_right: pd.merge(df_left, df_right, on=key_columns, how=join_as), dfs)
    return merged_df


def merge_dataframe_by_index(dfs: List[pd.DataFrame], key_indices=Optional[str, List[str]], join_as: str=['left', 'right', 'inner', 'outer']
                            )-> pd.DataFrame
    if key_indices:
        merged_df = reduce(df_left, df_right: pd.merge(df_left.set_index(key_indices), 
                                                       df_right.set_index(key_indices), left_index=True, right_index=True,
                                                       how=join_as), dfs)
    else:
        merged_df = reduce(df_left, df_right: pd.merge(df_left, df_right, left_index=True, right_index=True, how=join_as), dfs)
    return merged_df


def explode_column_into_columns(df, pd.DataFrame, col_to_explod: str, new_colums: List[str], 
                                replace_missing=None, keep_raw_col: bool=False):
    col_values_to_explode = list(zip_longest(*df[col_to_expand].tolist(), fillvalue=fillvalue))
    if len(col_values_to_explode) != len(new_colums):
        logger.debug(f"{len(new_columns)} new columns doesn't match expand from the current {col_to_expand} values!")

    for new_col, col_value in zip(new_colums, col_values_to_explode):
        df[new_col] = col_value
    if keep_raw_col:
        return df
    return df.drop(columns=col_to_expand)


# TODO: for all the plot options, shall we consider 1) return the plt out 2) store as pig
class DataFrameSummary:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df_ = df

    def get_metadata(self, metadata: Dict[Any, Any], groupby_cols: Union[str, List[str]], value2metrics: Dict[str, str]
                    ) -> Dict[str, Any]:
        metadata['summary'] = self.get_summary()
        for col in groupby_cols:
            metadata[f'{col}_dist'] = self.get_group_info(col, groupby_cols, {})
        return metadata

    def get_group_info(self, groupby_cols: Union[str, List[str]], value2metrics: Dict[str, str])-> pd.DataFrame:
        if isinstance(groupby_cols, str):
            groupby_cols = [groupby_cols]

        df_metrics = []
        for value_col, metric in value2metrics.items():
            if value_col not in self.df_.columns:
                raise VauleError(f'Unable to find column-{value_col} in dataframe!')
            elif metric is None:
                raise VauleError(f'Define a metric to calculate, e.g. sum, count, nunique, etc')
  
            cal_metrics = self.df_.groupby(groupby_cols).agg({value_col: metric}).reset_index()
            cal_metrics.columns = groupby_cols + [metric]
            df_metrics.append(cal_metrics)
        return merge_dataframes_by_columns(df_metrics, join_on=groupby_cols, join_as="inner")

    def get_columns_w_missing_value(self, missing_limit: int=.9) -> List[str]:
        """
        Checks which columns have over specified percentage of missing values
        Returns columns as a list
        """
        percent_missing = self.df_.isnull().mean()
        series = percent_missing[percent_missing > missing_limit]
        columns = series.index.to_list()
        return columns


    def drop_columns_w_many_nans(self, missing_limit: int=.9) -> pd.DataFrame:
        """
        Drops the columns whose missing value is bigger than missing percentage
        """
        cols_with_missing = self.view_columns_w_many_nans(missing_percent=missing_limit)
        list_of_cols = cols_with_missing.index.to_list()
        return self.df_.drop(columns=list_of_cols)

    def get_dimension(self, axis=None) -> int:
        dimensions = self.df_.shape
        if axis == 0:
            return dimensions[0]
        elif axis == 1:
            return dimensions[1]
        else:
            return dimensions

    def get_rescaled(self, norm_type = None, colnames = None, rrange = (0,1)) -> pd.DataFrame:
        if norm_type == "range":
            scaler = MinMaxScaler(feature_range = rrange)
        elif norm_type == "standard":
            scaler = StandardScaler()
        elif norm_type == "binary":
            scaler == Binarizer(threshold=0.0)
        else:
            scaler = Normalizer()
        if colnames is None:
            colnames = self.df_.columns
        df_scaled = pd.DataFrame(scaler.fit_transform(self.df_), columns=colnames)
        return df_scaled

    def get_summary(self) -> pd.DataFrame:
        return self.df_.describe()

    def get_corrlelations(self):
        return self.df_.corr(method='pearson')

    def get_skewness(self):
        return self.df_.skew()

    def get_histogram(self, colnums=[]):
        if colnums:
            self.df_[colnums].hist()
        else:
            self.df_.hist()
        pyplot.show()
        pyplot.close()

    def get_density_plot(self, colnum):
        num_dimensions = self.get_dimension(axis = 1)
        rownum = int(np.ceil(float(num_dimensions)/colnum))
        self.df_.plot(kind='density', subplots=True, layout=(rownum, colnum), sharex=False)
        pyplot.show()
        pyplot.close()

    def get_boxplot(self, colnum):
        num_dimensions = self.get_dimension(axis = 1)
        rownum = int(np.ceil(float(num_dimensions)/colnum))
        self.df_.plot(kind='box', subplots=True, layout=(rownum, colnum), sharex=False, sharey=False)
        pyplot.show()
        pyplot.close()

    def get_heatmap(self):
        corr = self.get_corrlelations()
        num_dimensions = self.get_dimension(axis = 1)
        names = self.df_.columns.values
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, num_dimensions, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        pyplot.show()
        pyplot.close()

    def get_scatter_matrix(self):
        scatter_matrix(self.df_)
        pyplot.show()
        pyplot.close()
