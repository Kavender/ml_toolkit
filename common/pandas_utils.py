from typing import List, Dict, Union, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from functools import reduce
from matplotlib import pyplot as plt
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


def transform_categorical_to_numerical(df: pd.DataFrame, categorical_numerical_mapping: Dict[str, Union[Dict, Callable]]) -> pd.DataFrame:
    """
    Transforms categorical columns to numerical columns based on the provided mapping.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - categorical_numerical_mapping (Dict[str, Union[Dict, Callable]]): A dictionary mapping column names to dictionaries or functions
        that define the transformation from categorical to numerical values.

    Returns:
    - pd.DataFrame: A new DataFrame with the specified categorical columns transformed to numerical columns.
    """
    transformers = {k: convert_categorical_to_numeric(v)
                    for k, v in categorical_numerical_mapping.items()}
    new_df = df.copy()
    for col, transformer in transformers.items():
        if col not in new_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        new_df[col] = new_df[col].map(transformer).astype('int64')
    return new_df


def convert_categorical_to_numeric(mapping: Union[Dict, Callable]) -> Callable:
    """
    Creates a transformer function based on the provided mapping.

    Parameters:
    - mapping (Union[Dict, Callable]): A dictionary or function that defines the transformation from categorical to numerical values.

    Returns:
    - Callable: A function that can be used to transform categorical values to numerical values.
    """
    if callable(mapping):
        return mapping
    elif isinstance(mapping, dict):
        return lambda x: mapping.get(x, -1)  # Default to -1 for unknown categories
    else:
        raise ValueError("Mapping must be a dictionary or a callable.")


def validate_required_fields(df: pd.DataFrame, required_cols: Union[str, List[str]]) -> bool:
    """
    Validates if the DataFrame contains the required columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be validated.
    - required_cols (Union[str, List[str]]): The columns that are expected in the DataFrame.

    Returns:
    - bool: True if all required columns are present. Otherwise, it raises a ValueError.

    Raises:
    - ValueError: If any of the required columns are missing from the DataFrame.
    """
    
    if isinstance(required_cols, str):
        required_cols = [required_cols]

    for col_name in required_cols:
        if col_name not in df.columns:
            raise ValueError(f"column- {col_name} not found in dataset. Make sure to set column {col_name} to the correct column - one of {', '.join(df.columns)}.")

    return True


def merge_dataframes_by_columns(
    dfs: List[pd.DataFrame], 
    key_columns: Union[str, List[str]], 
    join_as: str = 'inner'
) -> pd.DataFrame:
    """
    Merge multiple DataFrames on common columns.

    This function merges multiple DataFrames based on the specified key columns. 
    All DataFrames should have the key columns with the same names.

    Parameters:
    - dfs (List[pd.DataFrame]): A list of DataFrames to merge.
    - key_columns (Union[str, List[str]]): The column(s) to use as keys for merging.
    - join_as (str): The type of merge to perform; one of 'left', 'right', 'inner', or 'outer'. Default is 'inner'.

    Returns:
    - pd.DataFrame: The merged DataFrame.

    Reference:
    - https://stackoverflow.com/questions/23668427/pandas-three-way-joining-multiple-dataframes-on-columns
    """
    merged_df = reduce(
        lambda df_left, df_right: pd.merge(df_left, df_right, on=key_columns, how=join_as), 
        dfs
    )
    return merged_df


def merge_dataframe_by_index(
    dfs: List[pd.DataFrame], 
    key_indices: Union[str, List[str]] = None, 
    join_as: str = 'inner'
) -> pd.DataFrame:
    """
    Merge multiple DataFrames on common indices.

    This function merges multiple DataFrames based on the specified key indices. 
    All DataFrames should have the key indices with the same names.

    Parameters:
    - dfs (List[pd.DataFrame]): A list of DataFrames to merge.
    - key_indices (Union[str, List[str]], optional): The index(es) to use as keys for merging. Default is None.
    - join_as (str): The type of merge to perform; one of 'left', 'right', 'inner', or 'outer'. Default is 'inner'.

    Returns:
    - pd.DataFrame: The merged DataFrame.
    """
    if key_indices:
        merged_df = reduce(
            lambda df_left, df_right: pd.merge(
                df_left.set_index(key_indices), 
                df_right.set_index(key_indices), 
                left_index=True, 
                right_index=True,
                how=join_as
            ), 
            dfs
        )
    else:
        merged_df = reduce(
            lambda df_left, df_right: pd.merge(
                df_left, 
                df_right, 
                left_index=True, 
                right_index=True, 
                how=join_as
            ), 
            dfs
        )
    return merged_df


def concatenate_columns(df: pd.DataFrame, columns_to_concatenate: List[str], target_column: str, 
                        separator: Optional[str] = ' ') -> pd.DataFrame:
    """
    Concatenates multiple text columns in a DataFrame to create a final target feature column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns_to_concatenate (List[str]): List of column names to concatenate.
    - target_column (str): The name of the target feature column where the concatenated values will be stored.
    - separator (Optional[str], default=' '): The separator used between concatenated values.

    Returns:
    - pd.DataFrame: DataFrame with the concatenated feature column.

    """
    missing_columns = set(columns_to_concatenate).difference(df.columns)
    if len(missing_columns) > 0:
        raise ValueError(f"Cannot find column '{missing_columns}' in DataFrame.")

    df.fillna({col: "" for col in columns_to_concatenate}, inplace=True)
    df[target_column] = df[columns_to_concatenate].apply(lambda row: separator.join(row.values.astype(str)), axis=1)
    return df


def explode_column_into_columns(df: pd.DataFrame, col_to_explode: str, new_columns: List[str], 
                                replace_missing: Any = None, keep_raw_col: bool = False) -> pd.DataFrame:
    """
    Explodes the unique values in one column into multiple columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - col_to_explode (str): The name of the column to explode.
    - new_columns (List[str]): The names of the new columns to create.
    - replace_missing (Any, optional): The value to use to replace missing values. Default is None.
    - keep_raw_col (bool, optional): Whether to keep the original column. Default is False.

    Returns:
    - pd.DataFrame: The DataFrame with the exploded column.
    """
    # Check if the length of any list in col_to_explode doesn't match the length of new_columns
    len_check = df[col_to_explode].apply(len) != len(new_columns)
    if len_check.any():
        raise ValueError(f"The length of the lists in column '{col_to_explode}' must match the length of new_columns.")

    # Convert lists in col_to_explode to individual Series objects
    exploded_series = df[col_to_explode].apply(pd.Series)
    exploded_series.columns = new_columns

    # Replace missing values if any
    exploded_series.fillna(replace_missing, inplace=True)
    
    # Concatenate the exploded Series with the original DataFrame
    if keep_raw_col:
        return pd.concat([df, exploded_series], axis=1)
    else:
        return pd.concat([df.drop(columns=col_to_explode), exploded_series], axis=1)


def write_to_csv(df: pd.DataFrame, filename: str, header: Optional[bool] = None) -> None:
    """
    Write results (the DataFrame) to a CSV file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to write to CSV.
    - filename (str): The name of the file to write.
    - header (Optional[bool]): Whether to write the header. 
        If None, the header is written if the file does not exist.

    Raises:
    - ValueError: If the directory does not exist and cannot be created.
    - PermissionError: If the file cannot be written due to permissions.
    """
    file_path = Path(filename)
    directory = file_path.parent
    if not directory.exists():
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Unable to create directory: {directory}. {str(e)}")

    if header is None:
        header = not file_path.exists()

    try:
        df.to_csv(filename, mode="a" if file_path.exists() else "w", header=header, index=False)
    except PermissionError:
        raise PermissionError(f"Permission denied: Unable to write to {filename}")
    except Exception as e:
        raise ValueError(f"An error occurred while writing to {filename}. {str(e)}")



class MetadataExtractor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df_ = df
    
    def get_dimension(self, axis=None) -> int:
        dimensions = self.df_.shape
        if axis == 0:
            return dimensions[0]
        elif axis == 1:
            return dimensions[1]
        else:
            return dimensions

    def get_metadata(self, metadata: Dict[Any, Any], groupby_cols: Union[str, List[str]], 
                     value2metrics: Dict[str, str]
                    ) -> Dict[str, Any]:
        metadata['summary'] = self.get_summary()
        for col in groupby_cols:
            metadata[f'{col}_dist'] = self.get_group_info(col, groupby_cols, value2metrics)
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
    
    def get_summary(self) -> pd.DataFrame:
        return self.df_.describe()

    def get_corrlelations(self):
        return self.df_.corr(method='pearson')

    def get_skewness(self):
        return self.df_.skew()


class DataTransformer:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

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


class DataVisualizer:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def save_or_show_plot(self, filepath: Optional[str] = None) -> None:
        """Saves or displays the current plot based on the filepath parameter.

        Parameters:
        - filepath (Optional[str]): The path to save the plot. If None, the plot is displayed.
        """
        if filepath:
            plt.savefig(filepath)
        else:
            plt.show()
        plt.close()

    def get_histogram(self, colnums: Optional[List[str]] = None, filepath: Optional[str] = None) -> None:
        if colnums:
            self.df[colnums].hist()
        else:
            self.df.hist()
        self.save_or_show_plot(filepath)

    def get_density_plot(self, colnum: int, filepath: Optional[str] = None) -> None:
        num_dimensions = self.df.shape[1]
        rownum = int(np.ceil(float(num_dimensions)/colnum))
        self.df.plot(kind='density', subplots=True, layout=(rownum, colnum), sharex=False)
        self.save_or_show_plot(filepath)
    

    def get_boxplot(self, colnum: int, filepath: Optional[str] = None) -> None:
        num_dimensions = self.df.shape[1]
        rownum = int(np.ceil(float(num_dimensions)/colnum))
        self.df.plot(kind='box', subplots=True, layout=(rownum, colnum), sharex=False, sharey=False)
        self.save_or_show_plot(filepath)

    def get_heatmap(self, filepath: Optional[str] = None) -> None:
        corr = self.df.corr(method='pearson')
        num_dimensions = self.df.shape[1]
        names = self.df.columns.values
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, num_dimensions, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        self.save_or_show_plot(filepath)
    
    def get_scatter_matrix(self, filepath: Optional[str] = None) -> None:
        scatter_matrix(self.df)
        self.save_or_show_plot(filepath)


class DataFrameSummary:
    def __init__(self, df: pd.DataFrame) -> None:
        self.metadata_extractor = MetadataExtractor(df)
        self.data_transformer = DataTransformer(df)
        self.data_visualizer = DataVisualizer(df)