import pandas as pd
from pyhard.measures import RegressionMeasures
from pyhard.regression import RegressorsPool
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_pyhard_df(X, y):
    df = pd.DataFrame(X)
    df['y'] = y
    return df

def consolidate_data(*dataframes):
    
    # Create a list to store all dataframes
    dfs_to_concat = []
    
    # Process each dataframe
    for i, df in enumerate(dataframes):
        # If it's the first dataframe (index 0), reset the index and drop the 'index' column
        if i == 0:
            dfs_to_concat.append(df.reset_index().drop(columns=['index']))
        else:
            # For all other dataframes, just add them as is
            dfs_to_concat.append(df)
    
    # Concatenate all dataframes horizontally
    all_consolidated_df = pd.concat(dfs_to_concat, axis=1)
    
    return all_consolidated_df

def compute_regression_measures(df: dict, measures_list: list = None) -> tuple:
    if not measures_list:
        measures_dict = {
            'C4': 'collective_feature_efficiency',
            'L1': 'linear_absolute_error',
            'S1': 'output_distribution',
            'S2': 'input_distribution',
            'S3': 'error_nn_regressor',
            'FO': 'feature_outlier',
            'POT': 'target_outlier',
            'HB': 'histogram_bin',
            'EDS': 'stump_error',
            'EZR': 'zero_rule_error',
            'DS': 'disjunct_size',
            'TD_P': 'tree_depth_pruned',
            'TD_U': 'tree_depth_unpruned',
            'D': 'density',
            'CC': 'clustering_coefficient'
        }
        measures_list = list(measures_dict.keys())
    
    df_ = df.copy()
    regression_measures = RegressionMeasures(df_)
    
    # Create a custom wrapper for error_nn_regressor to handle shape mismatch
    original_error_nn_regressor = regression_measures.error_nn_regressor
    
    def safe_error_nn_regressor(n_neighbors=5):
        try:
            return original_error_nn_regressor(n_neighbors)
        except ValueError as e:
            if "operands could not be broadcast together" in str(e):
                logger.warning(f"Shape mismatch in error_nn_regressor: {str(e)}")
                # Return a default value for all instances
                return np.ones(len(df_)) * 0.5
            else:
                raise
    
    # Replace the method with our safe version
    regression_measures.error_nn_regressor = safe_error_nn_regressor
    
    # Calculate all measures
    regression_measures_df = regression_measures.calculate_all(measures_list=measures_list)
    return regression_measures_df

def create_pool_regressors(df: pd.DataFrame) -> RegressorsPool:
    pool_regressors = RegressorsPool(df)
    return pool_regressors

def compute_algorithm_performance(pool_regressors):
    performances_df = pool_regressors.run_all(
        hyper_param_optm=False, 
        n_folds=5, 
        n_iter=1,
        algo_list=None
    )
    return performances_df

def compute_instance_hardness(pool_regressors):
    instance_hardness = pd.DataFrame({\
        'instance_hardness': pool_regressors.estimate_ih()
    })\
        .assign(
            inverse_instance_hardness = lambda x: 1 - x['instance_hardness'],
            normalized_instance_hardness = lambda x: x['instance_hardness'] / x['instance_hardness'].max(),
            normalized_inverse_instance_hardness = \
                lambda x: x['inverse_instance_hardness'] / x['inverse_instance_hardness'].max()
        )
    return instance_hardness

def get_instance_hardness(X, y):
    df = build_pyhard_df(X, y)
    pool_regressors = create_pool_regressors(df)
    performances_df = compute_algorithm_performance(pool_regressors)
    regression_measures_df = None #compute_regression_measures(df)
    instance_hardness_df = compute_instance_hardness(pool_regressors)
    all_data = consolidate_data(df, regression_measures_df, performances_df, instance_hardness_df)
    return all_data

def remove_prefix(column_name: str) -> str:
    return column_name.replace('feature_', '')

def process_name(df: pd.DataFrame) -> pd.DataFrame:
    # Remove the prefix feature_ from the column names
    df_ = df.copy()
    renamed_columns = list()
    for col in df_.columns:
        if str(col).startswith('feature_'):
            renamed_columns.append(remove_prefix(col))
        else:
            renamed_columns.append(col)
    df_.columns = renamed_columns
    return df_

def get_hardness_measures(X, y, measures_list: dict = None):
    df = build_pyhard_df(X, y)
    hardness_measures = compute_regression_measures(df, measures_list)
    all_data = consolidate_data(df, hardness_measures)
    processed_named_data = process_name(all_data)
    return processed_named_data 