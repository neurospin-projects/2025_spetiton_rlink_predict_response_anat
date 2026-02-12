# System
import sys
import os
import os.path
import time
import json
from datetime import datetime


# Scientific python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from statsmodels.stats.multitest import multipletests

# Univariate statistics
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import statsmodels.stats.api as sms

# from itertools import product

# Models
from sklearn.base import clone
from sklearn.decomposition import PCA
import sklearn.linear_model as lm
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.neural_network import MLPClassifier

# Metrics
import sklearn.metrics as metrics

# Resampling
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
#from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import sklearn.linear_model as lm
from sklearn.compose import ColumnTransformer

################################################################################
# %% DataFrame utils

import pandas as pd

def expand_key_value_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Expands a column in a DataFrame containing key-value pairs into separate columns.

    The column should contain strings formatted as 'key1-value1_key2-value2_...' 
    where each key-value pair is separated by '_' and keys are separated from values by '-'.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame.
    col : str
        The name of the column in df containing the key-value strings.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame with the original column replaced by separate columns for each key.

    Example:
    --------
    >>> df = pd.DataFrame([
    ...     ["logistic", "fold-1_size-0.1", 0.5],
    ...     ["logistic", "fold-1_size-0.9", 0.5],
    ...     ["logistic", "fold-2_size-0.1", 0.5],
    ...     ["logistic", "fold-2_size-0.0", 0.5],
    ... ], columns=["model", "params", "auc"])
    >>> expand_key_value_column(df, "params")
       model  fold  size  auc
    0  logistic     1   0.1  0.5
    1  logistic     1   0.9  0.5
    2  logistic     2   0.1  0.5
    3  logistic     2   0.0  0.5
    """
    # Split key-value pairs
    kv_split = df[col].str.split('_').apply(
        lambda items: {k: v for k, v in (item.split('-') for item in items)}
    )
    
    # Convert list of dicts to DataFrame
    kv_df = pd.DataFrame(kv_split.tolist())

    # Convert to appropriate types if possible
    kv_df = kv_df.apply(pd.to_numeric, errors='ignore')

    # Reset index
    df = df.reset_index(drop=True)
    
    # Combine with original DataFrame (excluding the original key-value column)
    df_expanded = pd.concat([df.drop(columns=[col]), kv_df], axis=1)
    
    return df_expanded


def describe_categorical(df):
    """Describes categorical variables in a pandas DataFrame by counting
    occurrences of each category level.

    Parameters
    ----------
    df : DataFrame
        A pandas DataFrame containing categorical variables.

    Returns
    -------
    DataFrame
        A DataFrame with counts of each category level for each variable.
    Example
    -------
    Example:
    >>> df = pd.DataFrame({
    ...     'A': ['cat', 'dog', 'cat', 'bird'],
    ...     'B': ['dog', 'cat', 'fish', 'cat']
    ... })
    >>> print(describe_categorical(df))
         bird  cat  dog  fish
    A      1    2    1     0
    B      0    2    1     1
    """
        # If input is a Series, convert it to a DataFrame with a single column
    if isinstance(df, pd.Series):
        df = df.to_frame('Series')
        
    # Initialize an empty DataFrame to store the result
    result_df = pd.DataFrame()

    # Get all unique categories from all categorical columns
    unique_categories = sorted(df.stack().unique())

    # Iterate over each column in the input DataFrame
    for column in df.columns:
        # Count the occurrences of each category in the current column
        value_counts = df[column].value_counts().reindex(unique_categories, fill_value=0)
        # Add the counts as a new row to the result DataFrame
        result_df[column] = value_counts

    # Transpose the result DataFrame to have variables as rows and categories as columns
    result_df = result_df.T

    return result_df


################################################################################
# %% Read data utils

def get_y(data, target_column, remap_dict=None, print_log=print):
    """_summary_

    Parameters
    ----------
    data : _type_
        _description_
    target_column : _type_
        _description_
    remap_dict: dict
        remap target, defualts None
    print_log : _type_, optional
        _description_, by default print

    Returns
    -------
    _type_
        _description_

    Yields
    ------
    _type_
        _description_
    """
    print_log('\n# y (target)\n"%s", counts:' % target_column)
    print_log(describe_categorical(data[target_column]))

    if remap_dict:
        y = data[target_column].map(remap_dict)
        print_log('After remapping, counts:')
        print_log(describe_categorical(y))
    else:
        y = data[target_column]
    return y


def get_X(data, input_columns, print_log=print):
    """Get input Data. Perform dummy codings for categorical variables

    Parameters
    ----------
    data : DataFrame
        Input dataFrame
    input_columns : list
        input columns
    print_log: callable function, default print

    Returns
    -------
        pd.DataFrame: input Data
    """
    X = data[input_columns]
    ncol = X.shape[1]

    print_log('\n# X (Input data)')
    print_log(X.describe(include='all').T)

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        print_log("\nCategorical columns:", categorical_cols)
        for v in categorical_cols:
            print_log(v, describe_categorical(data[v]))
        X = pd.get_dummies(X, dtype=int)
        print_log('\nAfter coding')
        print_log(X.describe(include='all').T)
        print_log('%i dummies variable created' %  (X.shape[1] - ncol))

    return X


def get_residualizer(data, X, residualization_columns, print_log=print):
    """Residualiser

    Parameters
    ----------
    data : DataFrame
        input DataFrame
    X, : numpy Array
    residualization_columns : list of columns
        residualization variables
    print_log : callable, optional
        print function, by default print

    Returns
    -------
    Array, ResidualizerEstimator, str
        Array 
    """
    from mulm.residualizer import Residualizer
    from mulm.residualizer import ResidualizerEstimator
    
    print_log('\n# Residualization')
    # Residualizer
    residualization_formula = "+".join(residualization_columns)
    residualizer = Residualizer(data=data, formula_res=residualization_formula)

    # Extract design matrix and pack it with X
    Z = residualizer.get_design_mat(data=data)
    residualizer_estimator = ResidualizerEstimator(residualizer)
    
    # Repack Z with X
    X = residualizer_estimator.pack(Z, X)

    print_log(residualization_formula)
    print_log("Z.shape:", Z.shape)
    
    return X, residualizer_estimator, residualization_formula


def group_by_roi(feature_columns):
    """Group input columns by ROI name, removing optional 'Left' or 'Right' prefixes
    and known suffixes like '_GM_Vol' or '_CSF_Vol'.
    Returns a dictionary where keys are ROI names and values are lists of column names.
    
    Args:
        feature_columns (list): List of column names to group.
    Returns:
        dict: Dictionary with ROI names as keys and lists of column names as values.
    
    Exemple:
        >>> feature_columns_ = [
        ...     "Right Hippocampus_GM_Vol",
        ...     "Left Hippocampus_GM_Vol",
        ...     "Right Hippocampus_CSF_Vol",
        ...     "Left Hippocampus_CSF_Vol",
        ...     "3rd Ventricle_GM_Vol",
        ...     "4th Ventricle_GM_Vol"
        ... ]
        >>> result = group_by_roi(feature_columns_)
        >>> print(result)
        {'Hippocampus': ['Right Hippocampus_GM_Vol', 'Left Hippocampus_GM_Vol', 'Right Hippocampus_CSF_Vol', 'Left Hippocampus_CSF_Vol'], '3rd Ventricle': ['3rd Ventricle_GM_Vol'], '4th Ventricle': ['4th Ventricle_GM_Vol']}
        """
    import re
    from collections import defaultdict
    roi_dict = defaultdict(list)

    for col in feature_columns:
        # Match optional 'Left' or 'Right' prefix
        prefix_match = re.match(r'^(Left|Right)\s+', col)
        base = col

        if prefix_match:
            base = col[len(prefix_match.group(0)):]  # Remove prefix

        # Remove known suffixes if present
        roi_name = re.sub(r'_(GM|CSF)_Vol$', '', base)

        roi_dict[roi_name].append(col)

    return dict(roi_dict)

################################################################################
# %% Python utils

from itertools import product

def create_print_log(log_filename=None):
    """
    Creates and returns a print_log function that logs messages to a file if specified.

    The returned function, `print_log`, will print messages to a specified log file if 'log_filename'
    is present in the config dictionary. Otherwise, it will print messages to the standard output.

    Parameters:
    -----------
    config : dict
        A dictionary containing configuration settings. It should contain a key 'log_filename'
        with the path to the log file if logging to a file is desired.

    Returns:
    --------
    function
        A function that prints messages to the log file or standard output based on the config.

    Example:
    --------
    >>> print_log = create_print_log(log_filename='log.log')
    >>> print_log("This will be written to the log file.")
    """
    def print_log(*args):
        """
        Prints the provided arguments to the log file specified in the config or to the standard output.

        Parameters:
        -----------
        *args : list
            Variable length argument list containing the items to be printed.
        """
        if log_filename is not None:
            with open(log_filename, "a") as f:
                print(*args, file=f)
        else:
            print(*args)

    return print_log


def dict_cartesian_product(*dicts):
    """
    Compute the Cartesian product of multiple dictionaries.

    This function takes multiple dictionaries and returns a new dictionary where each key is a tuple
    representing the Cartesian product of the keys from the input dictionaries, and each value is a
    tuple of the corresponding values from the input dictionaries. If a key/value in the input dictionaries
    are a tuples, they are unpacked in the resulting dictionary.

    Parameters:
    *dicts : dict
        Variable number of dictionaries for which the Cartesian product is to be computed.

    Returns:
    dict
        A dictionary where each key is a tuple of keys from the input dictionaries, and each value
        is a tuple of the corresponding unpacked values.

    Examples:
    >>> dict1 = {("a", "A"): "aA", ("b", "B"): "bB"}
    >>> dict2 = {1: (1, 10), 2: (2, 20)}
    >>> dict_cartesian_product(dict1, dict2)
    {('a', 'A', 1): ('aA', 1, 10),
    ('a', 'A', 2): ('aA', 2, 20),
    ('b', 'B', 1): ('bB', 1, 10),
    ('b', 'B', 2): ('bB', 2, 20)}
    """
    # Compute the Cartesian product of keys
    keys_product = product(*[d.keys() for d in dicts])
    
    # Create the resulting dictionary with unpacked values
    result = {}
    for key_tuple in keys_product:
        
        # Collect values corresponding to the keys in the tuple
        values = []
        keys = []
        for i, key in enumerate(key_tuple):
            
            # flatten values if needed
            value = dicts[i][key]
            if isinstance(value, tuple):
                values.extend(value)
            else:
                values.append(value)
            
            # flatten keys if needed
            if isinstance(key, tuple):
                keys.extend(key)
            else:
                keys.append(key)
                
        result[tuple(keys)] = tuple(values)

    return result


################################################################################
# %% Build jobs utils

# Permutation 0 is without permutations
def permutation(x, random_state=None):
    if random_state == 0:
        return(x)
    if random_state is not None:
        np.random.seed(seed=random_state)
    return np.random.permutation(x)

################################################################################
# %% job utils

import time
from joblib import Parallel, delayed
from joblib import cpu_count

def run_sequential(func, iterable_dict, memory=None,  verbose=0,
                   *args, **kwargs):
    """Run a function sequentially over items in a dictionary.
    Uses a simple for loop to apply the function to each item in the dictionary.

    Parameters
    ----------
    func : callable
        The function to execute sequentially. It should accept the values from `iterable_dict` as arguments.
        The function signature should be `func(*args, verbose=0)`.
        The `verbose` argument is optional and can be used for logging.
        The function should return a result that will be collected in a dictionary.
    iterable_dict : dict
        Dictionary where each value is a tuple of arguments to pass to `func`.
    memory : object, optional
        Placeholder for memory caching (not used in this function), by default None.
    verbose : int, optional
        Verbosity level. If > 0, prints elapsed time, by default 0.
    *args
        Additional positional arguments to pass to `func`.
    **kwargs
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    _type_
        _description_
    """

    start_time = time.time()
    res = {k:func(*v, verbose=verbose) for k, v in iterable_dict.items()}

    if verbose > 0:
        print('Sequential execution, Elapsed time: \t%.3f sec' %
              (time.time() - start_time))
    
    return res


def run_parallel(func, iterable_dict, memory=None, n_jobs=None, verbose=0,
                   *args, **kwargs):
    """
    Run a function in parallel over items in a dictionary.
    Uses Joblib's Parallel and delayed for parallel execution.

    Parameters
    ----------
    func : callable
        The function to execute in parallel.
        It should accept the values from `iterable_dict` as arguments.
        The function signature should be `func(*args, verbose=0)`.
        The `verbose` argument is optional and can be used for logging.
        The function should return a result that will be collected in a dictionary.
    iterable_dict : dict
        Dictionary where each value is a tuple of arguments to pass to `func`.
    memory : object, optional
        Placeholder for memory caching (not used in this function), by default None.
    n_jobs : int, optional
        Number of parallel jobs to run. If None, uses the number of physical CPU cores.
    verbose : int, optional
        Verbosity level. If > 0, prints elapsed time, by default 0.
    *args
        Additional positional arguments to pass to `func`.
    **kwargs
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    dict
        Dictionary mapping the original keys to the results returned by `func`.

    Example
    -------
    >>> iterable_dict = {'a': (1, 2), 'b': (3, 4)}
    >>> def func(x, y, verbose=0):
    ...     if verbose > 0:
    ...         print(f"Processing {x}, {y}")
    ...     return x + y
    >>> results = run_parallel(func, iterable_dict, n_jobs=2, verbose=)
    >>> print(results)
    Processing 1, 2
    Processing 3, 4
    {'a': 3, 'b': 7}
    """

    if not n_jobs:
        n_jobs = cpu_count(only_physical_cores=True)
        
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)

    start_time = time.time()
    res = parallel(delayed(func)(*v, verbose=verbose)
                   for k, v in iterable_dict.items())

    if verbose > 0:
        print('Parallel execution, Elapsed time: \t%.3f sec' %
              (time.time() - start_time))

    return {k:r for k, r in zip(iterable_dict.keys(), res)}


################################################################################
# %% Sklearn utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import BaseCrossValidator
import numpy as np

import json

def save_folds(folds, json_file):
    """
    Save the folds of a scikit-learn cross-validation to a JSON file.

    This function takes a list of folds, where each fold is a tuple of train and test indices,
    and saves them to a specified JSON file.

    Parameters:
    -----------
    folds : list of tuples
        A list of tuples where each tuple contains two lists or arrays: the train indices and the test indices.
    json_file : str
        The path to the JSON file where the folds will be saved.

    Example:
    --------
    >>> folds = [(np.array([0, 1, 2]), np.array([3, 4])), (np.array([2, 3, 4]), np.array([0, 1]))]
    >>> save_folds(folds, 'folds.json')
    """
    # Convert numpy arrays to lists for JSON serialization
    folds_list = [(train.tolist(), test.tolist()) for train, test in folds]

    with open(json_file, 'w') as f:
        json.dump(folds_list, f)

def load_folds(json_file):
    """
    Load the folds of a scikit-learn cross-validation from a JSON file.

    This function reads a JSON file containing the folds of a cross-validation and returns them
    as a list of tuples of train and test indices.

    Parameters:
    -----------
    json_file : str
        The path to the JSON file from which the folds will be loaded.

    Returns:
    --------
    list of tuples
        A list of tuples where each tuple contains two lists: the train indices and the test indices.

    Example:
    --------
    >>> folds = load_folds('folds.json')
    >>> print(folds)
    [[[0, 1, 2], [3, 4]], [[2, 3, 4], [0, 1]]]
    """
    with open(json_file, 'r') as f:
        folds_list = json.load(f)

    # Convert lists back to tuples
    folds = [(train, test) for train, test in folds_list]

    return folds


class PredefinedSplit(BaseCrossValidator):
    """
    A custom cross-validator that uses pre-defined train/test splits.

    This class allows you to use pre-defined splits for cross-validation in scikit-learn.
    It is useful when you have specific train/test indices that you want to use directly.

    Parameters:
    -----------
    predefined_splits : list of tuples
        A list of tuples where each tuple contains two arrays: the train indices and the test indices.
        Each tuple represents a single split of the data.
    """

    def __init__(self, predefined_splits=None, json_file=None):
        self.predefined_splits = predefined_splits
        
        if not self.predefined_splits:
            self.predefined_splits = load_folds(json_file)
            
    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features), optional
            Training data, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape (n_samples,), optional
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into train/test sets.

        Returns:
        --------
        int
            The number of splits, which is the number of pre-defined splits provided.
        """
        return len(self.predefined_splits)

    def split(self, X=None, y=None, groups=None):
        """
        Generates indices to split data into training and test sets.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features), optional
            Training data, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape (n_samples,), optional
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into train/test sets.

        Yields:
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        for train_idx, test_idx in self.predefined_splits:
            yield train_idx, test_idx

    def to_json(self, json_file):
        save_folds(self.predefined_splits, json_file)


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class Ensure2D(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # stateless transformer

    def transform(self, X):
        # Case 1: pandas Series (1D) → convert to DataFrame
        if isinstance(X, pd.Series):
            return X.to_frame()

        # Case 2: pandas DataFrame (2D) → return as-is
        if isinstance(X, pd.DataFrame):
            return X

        # Case 3: NumPy array
        X = np.asarray(X)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class LogisticRegressionTransformer(LogisticRegression):
    def transform(self, X):
        return Ensure2D().transform(self.decision_function(X))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class MeanTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return Ensure2D().transform(X.mean(axis=1))

    
class GroupFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that groups sets of features and applies a specified transformation to each group.

    This transformer allows for grouping of features and applying transformations such as PCA, mean, LDA,
    logistic regression, or any custom transformer to each group.

    Parameters
    ----------
    groups : dict
        A dictionary where keys are the names of the new features and values are lists of indices or column names
        representing the features to be grouped.
    transformer : str or estimator object
        The transformer to apply to each group. It can be a string like "pca", "mean", "lda", "logistic", "regression",
        or an object that implements fit and transform methods.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> # Sample data with 4 rows and 5 columns
    >>> data = np.array([[1,   2,  3,  4,  5],
                         [6,   7,  8,  9, 10],
                         [11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20]])
    >>> X_df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])
    >>> groups = {
    ...     'Group1': [0, 1],  # Grouping first and second features
    ...     'Group2': [2, 3, 4]  # Grouping third, fourth, and fifth features
    ... }
    >>> transformer = GroupFeatureTransformer(groups, transformer="mean")
    >>> X_transformed_df = transformer.fit_transform(X_df)
    >>> print(X_transformed_df)
    [[ 1.5  4. ]
     [ 6.5  9. ]
     [11.5 14. ]
     [16.5 19. ]]
    """
    def __init__(self, groups, transformer):
        self.groups = groups
        self.transformer = transformer

    def _get_transformer(self):
        if isinstance(self.transformer, str):
            if self.transformer == "pca":
                return PCA(n_components=1)
            elif self.transformer == "mean":
                return MeanTransformer()
            elif self.transformer == "lda":
                return LinearDiscriminantAnalysis()
            elif self.transformer == "logistic":
                return LogisticRegression()
            elif self.transformer == "regression":
                return LinearRegression()
            else:
                raise ValueError(f"Unknown transformer string: {self.transformer}")
        elif hasattr(self.transformer, 'fit') and hasattr(self.transformer, 'transform'):
            return self.transformer
        else:
            raise ValueError("transformer must be a string or an object with fit and transform methods")

    def fit(self, X, y=None):

        self.column_transformer_ = ColumnTransformer(
            [(name, self._get_transformer(), features) for name, features in self.groups.items()]
        )
        self.column_transformer_.fit(X, y)
        return self

    def transform(self, X):

        return self.column_transformer_.transform(X)


class Ensure2D(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # stateless transformer

    def transform(self, X):
        # Case 1: pandas Series (1D) → convert to DataFrame
        if isinstance(X, pd.Series):
            return X.to_frame()

        # Case 2: pandas DataFrame (2D) → return as-is
        if isinstance(X, pd.DataFrame):
            return X

        # Case 3: NumPy array
        X = np.asarray(X)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


def pipeline_split(pipeline, step=-1):
    """Split pipeline into two pipelines body[:step] and head[step:].
    
    Parameters
    ----------
    pipeline : Pipeline
        A scikit-learn Pipeline object.
    step : int
        The step where to split the pipeline
    Returns
    -------
    body : Pipeline
        A new Pipeline object containing "step" fist steps.
    head : Pipeline
        A new Pipeline object containing the remaining steps.
    """
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Estimator must be a Pipeline instance")


    body = Pipeline(pipeline.steps[:step])
    head = Pipeline(pipeline.steps[step:])
    
    return body, head
    
    
def pipeline_behead(pipeline):
    """Separate preprocessing transformers from the prediction head of a pipeline estimator.
    This function assumes that the last step of the pipeline is the prediction head
    (e.g., a classifier or regressor) and all previous steps are preprocessing steps.
    
    Parameters
    ----------
    pipeline : Pipeline
        A scikit-learn Pipeline object.
    Returns
    -------
    transformers : Pipeline
        A new Pipeline object containing only the preprocessing steps.
    prediction_head : object
        The last step of the original pipeline, which is the prediction head (e.g., classifier
        Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 1, 0, 1])
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('clf', LogisticRegression())
    ... ])
    >>> pipe.fit(X, y)
    Pipeline(steps=[('scaler', StandardScaler()), ('clf', LogisticRegression())])
    >>> transformers, prediction_head = pipeline_behead(pipe)
    >>> # Apply preprocessing to data
    >>> X_preprocessed = transformers.transform(X)
    >>> # Use prediction head on preprocessed data
    >>> prediction_head.predict(X_preprocessed)
    array([0, 0, 1, 1])
    >>> pipe.fit(X, y)
    >>> pipe.predict(X)
    array([0, 0, 1, 1])
    """
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Estimator must be a Pipeline instance")

    #pipeline = clone(pipeline)  # Clone the estimator to avoid modifying the original
    preprocessing_steps = pipeline.steps[:-1]
    # Get the last step (the predictor)
    predictor_name, prediction_head = pipeline.steps[-1]
    # Create a new pipeline with only the preprocessing steps
    transformers = Pipeline(preprocessing_steps)
    
    return transformers, prediction_head

    
def get_linear_coefficients(estimator):
    """
    Retrieve the linear coefficient(s) from a scikit-learn estimator or pipeline.

    Handles:
    - Estimator is a Pipeline: returns the coefficient of the last step.
    - Estimator or last step is a GridSearchCV: returns the coefficient of the best_estimator_.
    - If the final estimator has no 'coef_' attribute, returns None.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        A scikit-learn estimator, pipeline, or GridSearchCV object.

    Returns
    -------
    coef : np.ndarray or None
        The linear coefficient(s), or None if not available.

    Examples
    --------

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X = np.array([[0, 1], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> # Direct estimator
    >>> lr = LogisticRegression().fit(X, y)
    >>> get_linear_coefficients(lr).shape
    (1, 2)
    >>> # Pipeline
    >>> pipe = Pipeline([('clf', LogisticRegression())]).fit(X, y)
    >>> get_linear_coefficients(pipe).shape
    (1, 2)
    >>> # GridSearchCV
    >>> param_grid = {'C': [0.1, 1]}
    >>> grid = GridSearchCV(LogisticRegression(), param_grid, cv=2).fit(X, y)
    >>> get_linear_coefficients(grid).shape
    (1, 2)
    >>> # Pipeline + GridSearchCV
    >>> pipe = Pipeline([('clf', LogisticRegression())])
    >>> grid = GridSearchCV(pipe, {'clf__C': [0.1, 1]}, cv=2).fit(X, y)
    >>> get_linear_coefficients(grid).shape
    (1, 2)
    >>> # Non-linear estimator
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> tree = DecisionTreeClassifier().fit(X, y)
    >>> print(get_linear_coefficients(tree))
    None
    """

    while True:

        # Unwrap Pipeline
        if isinstance(estimator, Pipeline):
            estimator = estimator.steps[-1][1]  # get the last step's estimator

        # Unwrap GridSearchCV
        elif isinstance(estimator, GridSearchCV):
            estimator = estimator.best_estimator_

        # Check for coef_ attribute (used by linear models)
        elif hasattr(estimator, 'coef_'):
            return estimator.coef_

        else:
            return None


################################################################################
# %% Feature importance utils

def single_feature_classif(X_train, X_test, y_train, y_test):
    
    make2d = Ensure2D()
    lr = lm.LogisticRegression(fit_intercept=True, class_weight='balanced')
    aucs = np.array([metrics.roc_auc_score(y_test,
                    lr.fit(make2d.transform(X_train[:, j]), y_train).decision_function(make2d.transform(X_test[:, j]))) 
                    for j in range(X_train.shape[1])])
    return aucs

################################################################################
# %% Mapper: Fit and predict function used in parallel execution
# -----------------------------------------------------------



def fit_predict_binary_classif(estimator, X, y, train_idx, test_idx, **kwargs):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    estimator.fit(X_train, y_train)
    
    # 1. Predictions
    y_test_pred_lab = estimator.predict(X_test)
    y_test_pred_proba = estimator.predict_proba(X_test)[:, 1]  # Probability of the positive class
    if hasattr(estimator, 'decision_function'):
        y_test_pred_decision_function = estimator.decision_function(X_test)
    else:
        y_test_pred_decision_function = None # np.nan * np.ones_like(y_test_pred_proba)
    # Decision function for SVMs
    # y_proba = estimator.predict_log_proba(X[test])[:, 1

    # 2. Feature importance
    if isinstance(estimator, Pipeline):
        transformers, head = pipeline_behead(estimator)
        X_train = transformers.transform(X_train)
        X_test = transformers.transform(X_test)
        # prediction_head.fit(Xtr_train, y[train_idx])
    else:
        head = estimator
    
    # 2.1 Coefficients of the prediction head
    coefs_ = get_linear_coefficients(head)
    if coefs_ is not None:
        coefs_ = coefs_.squeeze()
    
    # 2.2 Forward models [Haufe 2014 Forward model] eq. 7:
    if hasattr(head, 'decision_function'):
        #estimator.decision_function(X_test)
        s = head.decision_function(X_test)
        forwd_ = np.dot(X_test.T, s)
    else:
        forwd_ = None
    
    ### 2.3 Individual feature predictive power
    feature_auc = single_feature_classif(X_train, X_test, y_train, y_test)

    # print(y_, y_[test_idx])
    return dict(train_idx=train_idx, test_idx=test_idx,
                y_test_pred_lab=y_test_pred_lab,
                y_test_pred_proba=y_test_pred_proba,
                y_test_pred_decision_function=y_test_pred_decision_function,
                y_test_true_lab=y_test,
                coefs=coefs_, forwd=forwd_, feature_auc=feature_auc,
                estimator=estimator)


################################################################################
# %% Reducers
#
# 1. Convert results into DataFrame
# 2. Compute metrics for prediction or feature importance

# 1. Convert result of (parrallel execution) into DataFrame on which we will compute
# metrics

def dict_to_frame(input_dict, keys, base_dict={}):
    """
    Convert a subset of `input_dict` into a one-row pandas DataFrame,
    optionally extended with fixed values from `base_dict`.

    Parameters
    ----------
    input_dict : dict
        A dictionary containing values (typically lists or scalars) to extract.
    keys : list of str
        Keys from `input_dict` to include in the output DataFrame.
    base_dict : dict, optional
        A dictionary of fixed values (scalars or lists) to include in all rows,
        by default {}.

    Returns
    -------
    pd.DataFrame
        A DataFrame constructed from `input_dict[keys]` and `base_dict`. The number of rows
        corresponds to the length of the values in `input_dict[keys[0]]` (assumed consistent).

    Examples
    --------
    >>> input_dict = dict(x1=[1, 2], x2=[10, 20])
    >>> dict_to_frame(input_dict, keys=['x1', 'x2'])
       x1  x2
    0   1  10
    1   2  20

    >>> base_dict = dict(model='regression', fold='1')
    >>> dict_to_frame(input_dict, keys=['x1', 'x2'], base_dict=base_dict)
       model      fold  x1  x2
    0  regression     1   1  10
    1  regression     1   2  20
    """
    output_dict = base_dict.copy()
    output_dict.update({k: input_dict[k] for k in keys})
    return pd.DataFrame(output_dict)


# 2.1 Classification metrics

class ClassificationScorer:
    
    def __init__(self,
                 y_true_lab="y_test_true_lab",
                 y_pred_lab="y_test_pred_lab",
                 y_pred_decision_function="y_test_pred_decision_function",
                 y_pred_proba="y_test_pred_proba",
                 metrics_names=['balanced_accuracy', 'roc_auc']):
    
        self.y_true_lab=y_true_lab
        self.y_pred_lab=y_pred_lab
        self.y_pred_decision_function=y_pred_decision_function
        self.y_pred_proba=y_pred_proba
        self.metrics_names=metrics_names


    def scores(self, y_true_lab, y_pred_lab, y_pred_decision_function, y_pred_proba):
        balanced_accuracy = metrics.balanced_accuracy_score(
            y_true_lab, y_pred_lab)
        # take decision_function when exists else proba
        roc_auc = metrics.roc_auc_score(
            y_true_lab,
            np.where(~np.isnan(y_pred_decision_function),
                        y_pred_decision_function, y_pred_proba))

        return balanced_accuracy, roc_auc

    def predictions_dict_toframe(self, predictions_dict, keys=['test_idx', 'y_test_pred_lab',
                                                        'y_test_pred_decision_function',
                                                        'y_test_pred_proba',
                                                        'y_test_true_lab']):
        
        predictions_df = pd.concat([dict_to_frame(input_dict=val_dict, keys=keys,
                                            base_dict={'model':mod, 'perm':perm, 'fold':fold})
                            for (mod, perm, fold), val_dict in predictions_dict.items()])
        return predictions_df

    def prediction_metrics(self, predictions_df):

        # Compute metrics per model and permutation, and folds
        predictions_metrics_df =  pd.DataFrame(
            [[mod, perm, fold] + list(self.scores(
                predictions_bymodbyperm_df[self.y_true_lab],
                predictions_bymodbyperm_df[self.y_pred_lab],
                predictions_bymodbyperm_df[self.y_pred_decision_function],
                predictions_bymodbyperm_df[self.y_pred_proba]))
            for (mod, perm, fold), predictions_bymodbyperm_df 
            in predictions_df.groupby(['model', 'perm', 'fold'])],
            columns=["model", "perm", 'fold'] + self.metrics_names)

        # Average accross folds
        predictions_metrics_df = \
            predictions_metrics_df.groupby(['model', 'perm']).mean(numeric_only=True).reset_index().sort_values('perm')

        return predictions_metrics_df

    def prediction_metrics_pvalues(self, predictions_metrics_df, permutation_col='perm', true_value='perm-000'):
        
        # Compute p-values and add it to predictions_metrics_df
        predictions_metrics_true_df = predictions_metrics_df[predictions_metrics_df[permutation_col] == true_value]
        #print(predictions_metrics_true_df)
        predictions_metrics_rnd_df = predictions_metrics_df[predictions_metrics_df[permutation_col] != true_value]

        if predictions_metrics_rnd_df.shape[0] > 0:
            #metrics_names = ["balanced_accuracy", "roc_auc"]
            predictions_metrics_pval_df = list()

            for mod, predictions_metrics_rnd_bymod_df in predictions_metrics_rnd_df.groupby('model'):
                nperms = predictions_metrics_rnd_bymod_df.shape[0]
                balanced_accuracy_h0_mean, roc_auc_h0_mean = predictions_metrics_rnd_bymod_df[self.metrics_names].mean(numeric_only=True)
                balanced_accuracy_h0_std, roc_auc_h0_std = predictions_metrics_rnd_bymod_df[self.metrics_names].std(numeric_only=True)
                balanced_accuracy_h0_pval, roc_auc_h0_pval = (predictions_metrics_rnd_bymod_df[self.metrics_names].values > 
                                                            predictions_metrics_true_df.loc[predictions_metrics_true_df['model']==mod, self.metrics_names].values).sum(axis=0) / nperms

                predictions_metrics_pval_df.append([mod,
                    balanced_accuracy_h0_pval, roc_auc_h0_pval,
                    balanced_accuracy_h0_mean, roc_auc_h0_mean, 
                    balanced_accuracy_h0_std, roc_auc_h0_std])

            predictions_metrics_pval_df = pd.DataFrame(predictions_metrics_pval_df, columns=['model',
                    'balanced_accuracy_h0_pval', 'roc_auc_h0_pval',
                    'balanced_accuracy_h0_mean', 'roc_auc_h0_mean', 
                    'balanced_accuracy_h0_std',  'roc_auc_h0_std'])

            predictions_metrics_true_df = pd.merge(predictions_metrics_true_df, predictions_metrics_pval_df, how='left')

            return predictions_metrics_true_df
        else:
            return None



# 2.2 Feature importance metrics
#
# Feature importance with coefficients of forward model
# Forward-based feature importance [Haufe 2014 Forward model](https://www.sciencedirect.com/science/article/pii/S1053811913010914)
# Forward coefficients: Cov(X, s) where s is the decision function of the prediction head
# s = prediction_head.decision_function(Xtr_train)
# Cov(X, s) = X^T s
# where X is the input data after preprocessing (transformers.fit_transform(Xtr_train, y[train_idx]))

def mean_sd_tval_pval_ci(betas_rep, m0=0):
    """_summary_

    Parameters
    ----------
    betas_rep : array n_repetitions x n_features
        repetition of parameters

    m0 : float
        mean under H0
    Returns
    -------
    _type_
        _description_
    """
    betas_m = np.mean(betas_rep, axis=0)      # sample mean
    betas_s = np.std(betas_rep, ddof=1, axis=0)  # sample standard deviation
    n = betas_rep.shape[0]             # sample size
    df = n - 1
    betas_tval = (betas_m - m0) / betas_s * np.sqrt(n)
    betas_tval_abs = np.abs(betas_tval)
    betas_pval = scipy.stats.t.sf(betas_tval_abs, df) * 2
    #betas_pval_bonf = multipletests(betas_pval, method='bonferroni')[1]
    #betas_pval_fdr = multipletests(betas_pval, method='fdr_bh')[1]
    
    # Critical value for t at alpha / 2:
    t_alpha2 = -scipy.stats.t.ppf(q=0.05/2, df=df, loc=m0)
    ci_low = betas_m - t_alpha2 * betas_s / np.sqrt(n)
    ci_high = betas_m + t_alpha2 * betas_s / np.sqrt(n)
 
    return betas_m, betas_tval, betas_tval_abs,\
        betas_pval, ci_low, ci_high

# labels features depending en model see: features_names[mod]
def stack_features_dicts(res_cv, models_features_names, importances=['coefs', 'forwd', 'feature_auc']):
    """Collect and stack into DataFrame output dictionaries: res_cv

    Parameters
    ----------
    res_cv : Dictionary
        indexed by model, permutation, and fold
        Items are dictionaries produced by fit_predict
    models_features_names : Dictionary
        key = models, values = features names.
        useful when models perform feature aggregation modifying input features
    importances : list of string, optional
        should match keys of res_cv, by default ['coefs', 'forwd', 'feature_auc']

    Returns
    -------
    _type_
        _description_
    """
    features_df = pd.concat([dict_to_frame(input_dict=val_dict,
        keys=importances,
        base_dict={'model':mod, 'perm':perm, 'fold':fold, 'feature':models_features_names[mod]})
        for (mod, perm, fold), val_dict in res_cv.items()])
    return features_df

def features_statistics(features_df, feature_name_col='feature',
                        feature_importance_cols={'coefs':0, 'forwd':0, 'feature_auc':0.5}):
    """Compute feature statistics.
    Group by 'model', 'perm', 'fold'
        for each feature 'feature'
            for all importance given 'feature_importance_cols'
                Compute statistics across fold

    Parameters
    ----------
    features_df : DataFrame 
        with 'model', 'perm', 'fold', 'feature' columns
    feature_name_col : str, optional
        column name of the feature, by default 'feature'
    feature_importance_cols : dict, optional
        importance columns with corresponding null hypothesis, by default {'coefs':0, 'forwd':0, 'feature_auc':0.5}

    Returns
    -------
    DataFrame
        For all 'model', 'perm', 'fold', 'feature', and 'importance',
        compute 'mean', 'tval', 'tval_abs', 'pval', 'ci_low' and 'ci_high'
    """

    # For all mod, perm concatenate values accross folds store if in a features[(mod, perm), feature_importance_col]
    from collections import defaultdict
    features = defaultdict(list)

    # For each feature importance statistic append across folds
    # dict[(mod, perm), feature_importance_col] = [vals0, vals1, ...]
    for (mod, perm, fold), df in features_df.groupby(['model', 'perm', 'fold']):
        df = df.set_index(feature_name_col)
        for feature_importance_col in feature_importance_cols.keys():
            features[(mod, perm), feature_importance_col].append(df[feature_importance_col]) 

    # Feature importance folds-wise statistics
    features_stats = list()
    for ((mod, perm), feature_importance_col), vals in features.items():
        #print(mod, perm, feature_importance_col, vals.shape)
        vals = pd.concat(vals, axis=1)
        for i in range(vals.shape[0]):
            features_stats.append([mod, perm, vals.index[i], feature_importance_col] +\
                list(mean_sd_tval_pval_ci(vals.iloc[i, :].values, m0=feature_importance_cols[feature_importance_col])))

    features_stats = pd.DataFrame(features_stats, columns=['model', 'perm', feature_name_col, 'importance', 'mean', 'tval', 'tval_abs', 'pval', 'ci_low', 'ci_high'])

    return features_stats


def features_statistics_pvalues(features_stats):
    stat_pval = {}
    # Split by models and compute corrected p-values
    for (mod, stat), df in features_stats.groupby(['model', 'importance']):
        #if mod == 'model-grpRoiLda+lrl2_resid-age+sex+site' and stat=='feature_auc':
        #if mod == 'model-grpRoiLda+lrl2_resid-age+sex+site' and stat=='forwd':
        #    break
        print(mod, stat)
        true_df = df[df['perm']=='perm-000'].copy().set_index('feature')
        rnd_df = df[df['perm']!='perm-000']
        
        # Corrected p-values
        true_df['pval_bonferroni'] = multipletests(true_df['pval'], method='bonferroni')[1]
        true_df['pval_fdr_bh'] = multipletests(true_df['pval'], method='fdr_bh')[1]

        # Statistics under H0
        mean_rnd = pd.concat([df_.set_index('feature')['mean'] for perm, df_ in rnd_df.groupby('perm')], axis=1)    
        true_df['mean_h0'] = mean_rnd.mean(axis=1)
        tval_rnd = pd.concat([df_.set_index('feature')['tval'] for perm, df_ in rnd_df.groupby('perm')], axis=1)
        true_df['tval_h0'] = tval_rnd.mean(axis=1)
        # if mod == 'model-grpRoiLda+lrl2_resid-age+sex+site' and stat=='forwd':
        #     break
        # #tval_true_ = true_df.loc[tval_rnd.index == "Hippocampus", 'tval']
        # tval_rnd_ = tval_rnd.loc[tval_rnd.index == "Hippocampus", :].values.ravel()
        # perms_ = np.array([int(perm.split("-")[1]) for perm, _ in rnd_df.groupby('perm')])
        # n_ok = np.where(tval_rnd_ >= 2.80743991428207)[0]
        # ok = np.where(tval_rnd_ < 2.80743991428207)[0]
        # perms_ = perms_[np.sort(list(np.random.choice(ok, 955, replace=False)) + list(np.random.choice(n_ok, 45, replace=False)))]
        # perms_df = pd.DataFrame(dict(perm=perms_))
        # perms_df.to_csv("permutations.csv", index=False)
        
        
        # Compute randomize p-values
        true_df['mean_pval_rnd'] = mean_rnd.gt(true_df['mean'], axis=0).sum(axis=1) / mean_rnd.shape[1]
        true_df['tval_pval_rnd'] = tval_rnd.gt(true_df['tval'], axis=0).sum(axis=1) / tval_rnd.shape[1]
        
        # Westfall & Young FWER cor p-values

        stat_max_rnd = tval_rnd.max(axis=0).values #stat_max_rnd.shape: (nperms, )
        stat_values = true_df['tval'].values # stat_values.shape: (nfeatures, )
        # Reshape for broadcasting: (nfeatures, 1) vs (1, nperms)
        nperms = stat_max_rnd.shape[0]
        pval_tmax_fwer = (stat_max_rnd > stat_values[:, None]).sum(axis=1) / nperms # pval_tmax_fwer.shape (nfeatures, )
        true_df['tval_pval_tmax_fwer'] = pd.Series(pval_tmax_fwer, index=true_df.index)

        true_df = true_df.reset_index()
        true_df.sort_values('pval', inplace=True)
        stat_pval[(mod, stat)] = true_df
    
    return stat_pval

################################################################################
# %% Models

mlp_param_grid = {"hidden_layer_sizes":
                    [
                    (100, ), (50, ), (25, ), (10, ), (5, ),       # 1 hidden layer
                    (100, 50, ), (50, 25, ), (25, 10,), (10, 5, ), # 2 hidden layers
                    (100, 50, 25, ), (50, 25, 10, ), (25, 10, 5, ), # 3 hidden layers
                    ],
                    "activation": ["relu"], "solver": ["sgd"], 'alpha': [0.0001]}


def make_models(n_jobs_grid_search, cv_val,
                scoring='accuracy',
                residualization_formula=None,
                residualizer_estimator=None,
                roi_groups=None):
    """Make models

    Parameters
    ----------
    n_jobs_grid_search : int
        Nb jobs for grd search
    scoring : str, optional
         'balanced_accuracy' 'roc_auc', by default 'accuracy'
    residualization_formula : str, optional
        if string is provided use residualization, and labels model with this
        string, by default None
    residualizer_estimator :  ResidualizerEstimator()
        The residualizer

    Returns
    -------
    dict
        Dictionary of models
    """
    
    # Param grd for MLP
    mlp_param_grid = {"hidden_layer_sizes":
                      [(100, ), (50, ), (25, ), (10, ), (5, ),         # 1 hidden layer
                       (100, 50, ), (50, 25, ), (25, 10,
                                                 # 2 hidden layers
                                                 ), (10, 5, ),
                          # 3 hidden layers
                       (100, 50, 25, ), (50, 25, 10, ), (25, 10, 5, )],
                      "activation": ["relu"], "solver": ["sgd"], 'alpha': [0.0001]}


    # Models models_backbones to be completed with residualizer, 
    models_backbones = {
        'model-lrl2cv':[
            preprocessing.StandardScaler(),
            # preprocessing.MinMaxScaler(),
            GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
                         {'C': 10. ** np.arange(-3, 1)},
                         cv=cv_val, n_jobs=n_jobs_grid_search, scoring=scoring)],

        'model-lrenetcv':[
            preprocessing.StandardScaler(),
            # preprocessing.MinMaxScaler(),
            GridSearchCV(estimator=lm.SGDClassifier(loss='log_loss',
                                                    penalty='elasticnet',
                                                    fit_intercept=False, class_weight='balanced'),
                         param_grid={'alpha': 10. ** np.arange(-1, 3),
                                     'l1_ratio': [.1, .5, .9]},
                         cv=cv_val, n_jobs=n_jobs_grid_search, scoring=scoring)],

        'model-svmrbfcv':[
            preprocessing.StandardScaler(),
            #preprocessing.MinMaxScaler(),
            GridSearchCV(svm.SVC(class_weight='balanced', probability=True),
                         # {'kernel': ['poly', 'rbf'], 'C': 10. ** np.arange(-3, 3)},
                         {'kernel': ['rbf'], 'C': 10. ** np.arange(-1, 2)},
                         cv=cv_val, n_jobs=n_jobs_grid_search, scoring=scoring)],

        'model-forestcv':[
            preprocessing.StandardScaler(),
            #preprocessing.MinMaxScaler(),
            GridSearchCV(RandomForestClassifier(random_state=1, class_weight='balanced'),
                         {"n_estimators": [10, 100]},
                         cv=cv_val, n_jobs=n_jobs_grid_search, scoring=scoring)],

        'model-gbcv':[
            preprocessing.StandardScaler(),
            #preprocessing.MinMaxScaler(),
            GridSearchCV(estimator=GradientBoostingClassifier(random_state=1),
                         param_grid={"n_estimators": [10, 100]},
                         cv=cv_val, n_jobs=n_jobs_grid_search, scoring=scoring)],

        'mlp_cv':[
            preprocessing.StandardScaler(),
            # preprocessing.MinMaxScaler(),
            GridSearchCV(estimator=MLPClassifier(random_state=1, max_iter=200, tol=0.01),
                         param_grid=mlp_param_grid,
                         cv=cv_val, n_jobs=n_jobs_grid_search)],
        
        'grpRoiLda+lrl2':[
            GroupFeatureTransformer(roi_groups,  "lda"),
            preprocessing.StandardScaler(),
            GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
                        {'C': 10. ** np.arange(-3, 1)},
                        cv=cv_val, n_jobs=5, scoring='balanced_accuracy')]
    }

    if residualization_formula:
        models = {model_name_prefix + "_resid-%s" % residualization_formula:\
            make_pipeline(* [residualizer_estimator] + model_steps)
                        for model_name_prefix, model_steps
                        in models_backbones.items()}

    else:
        models = {model_name_prefix:\
            make_pipeline(*model_steps)
                    for model_name_prefix, model_steps
                    in models_backbones.items()}
    
    return models

def make_models_orig(n_jobs_grid_search, cv_val,
                residualization_formula=None,
                residualizer_estimator=None):
    """_summary_

    Parameters
    ----------
    n_jobs_grid_search : int
        Nb jobs for grd search
    residualization_formula : str, optional
        if string is provided use residualization, and labels model with this
        string, by default None
    residualizer_estimator :  ResidualizerEstimator()
        The residualizer

    Returns
    -------
    dict
        Dictionary of models
    """
    
    # Param grd for MLP
    mlp_param_grid = {"hidden_layer_sizes":
                      [(100, ), (50, ), (25, ), (10, ), (5, ),         # 1 hidden layer
                       (100, 50, ), (50, 25, ), (25, 10,
                                                 # 2 hidden layers
                                                 ), (10, 5, ),
                          # 3 hidden layers
                       (100, 50, 25, ), (50, 25, 10, ), (25, 10, 5, )],
                      "activation": ["relu"], "solver": ["sgd"], 'alpha': [0.0001]}


    # Models models_backbones to be completed with residualizer, 
    models_backbones = {
        'model-lrl2cv':[
            preprocessing.StandardScaler(),
            # preprocessing.MinMaxScaler(),
            GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
                         {'C': 10. ** np.arange(-3, 1)},
                         cv=cv_val, n_jobs=n_jobs_grid_search)],

        'model-lrenetcv':[
            preprocessing.StandardScaler(),
            # preprocessing.MinMaxScaler(),
            GridSearchCV(estimator=lm.SGDClassifier(loss='log_loss',
                                                    penalty='elasticnet',
                                                    fit_intercept=False, class_weight='balanced'),
                         param_grid={'alpha': 10. ** np.arange(-1, 3),
                                     'l1_ratio': [.1, .5, .9]},
                         cv=cv_val, n_jobs=n_jobs_grid_search)],

        'model-svmrbfcv':[
            # preprocessing.StandardScaler(),
            preprocessing.MinMaxScaler(),
            GridSearchCV(svm.SVC(class_weight='balanced'),
                         # {'kernel': ['poly', 'rbf'], 'C': 10. ** np.arange(-3, 3)},
                         {'kernel': ['rbf'], 'C': 10. ** np.arange(-1, 2)},
                         cv=cv_val, n_jobs=n_jobs_grid_search)],

        'model-forestcv':[
            # preprocessing.StandardScaler(),
            preprocessing.MinMaxScaler(),
            GridSearchCV(RandomForestClassifier(random_state=1, class_weight='balanced'),
                         {"n_estimators": [10, 100]},
                         cv=cv_val, n_jobs=n_jobs_grid_search)],

        'model-gbcv':[
            preprocessing.MinMaxScaler(),
            GridSearchCV(estimator=GradientBoostingClassifier(random_state=1),
                         param_grid={"n_estimators": [10, 100]},
                         cv=cv_val, n_jobs=n_jobs_grid_search)],

    #     'mlp_cv':[
    #         # preprocessing.StandardScaler(),
    #         preprocessing.MinMaxScaler(),
    #         GridSearchCV(estimator=MLPClassifier(random_state=1, max_iter=200, tol=0.01),
    #                      param_grid=mlp_param_grid,
    #                      cv=cv_val, n_jobs=n_jobs_grid_search)]
    }

    if residualization_formula:
        models = {model_name_prefix + "_resid-%s" % residualization_formula:\
            make_pipeline(* [residualizer_estimator] + model_steps)
                        for model_name_prefix, model_steps
                        in models_backbones.items()}

    else:
        models = {model_name_prefix:\
            make_pipeline(*model_steps)
                    for model_name_prefix, model_steps
                    in models_backbones.items()}

    return models

