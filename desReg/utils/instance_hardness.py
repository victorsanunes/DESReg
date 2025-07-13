import pandas as pd
from pyhard.measures import RegressionMeasures
from pyhard.regression import RegressorsPool
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        measures_list = list(measures_dict.keys())
    
    df_ = df.copy()
    regression_measures = RegressionMeasures(df_)
    
    # Create a custom implementation of error_nn_regressor to fix the shape mismatch
    original_error_nn_regressor = regression_measures.error_nn_regressor
    
    def fixed_error_nn_regressor(n_neighbors=5):
        try:
            # Get the distance matrix
            dist_matrix = regression_measures.dist_matrix_gower
            
            # Create a NearestNeighbors object with the distance matrix
            nn = NearestNeighbors(n_neighbors=n_neighbors+1, metric='precomputed').fit(dist_matrix)
            
            # Get the indices of the k+1 nearest neighbors (including self)
            distances, indices = nn.kneighbors(dist_matrix)
            
            # Remove the first neighbor (self)
            indices = indices[:, 1:]
            
            # Calculate the mean of the target values for the k nearest neighbors
            nn_y_avg = np.array([np.mean(regression_measures.y_scaled[indices[i]]) for i in range(len(indices))])
            
            # Calculate the squared error
            S3 = (nn_y_avg - regression_measures.y_scaled) ** 2
            
            # Normalize and return
            return 1 - np.exp(-S3 / S3.std())
        except Exception as e:
            logger.error(f"Error in fixed_error_nn_regressor: {str(e)}")
            # Return a default value for all instances
            return np.ones(len(df_)) * 0.5
    
    # Alternative implementation that processes the flattened array correctly
    def alternative_error_nn_regressor(n_neighbors=5):
        try:
            # Get the distance matrix
            dist_matrix = regression_measures.dist_matrix_gower
            
            # Create a NearestNeighbors object with the distance matrix
            nn = NearestNeighbors(n_neighbors=n_neighbors+1, metric='precomputed').fit(dist_matrix)
            
            # Get the sparse matrix representation
            nn_arr = nn.kneighbors_graph(mode='distance').toarray()
            
            # Get the row and column indices of non-zero elements
            rows, cols = np.nonzero(nn_arr)
            
            # Create a dictionary to store neighbors for each instance
            instance_neighbors = {}
            for i in range(len(df_)):
                instance_neighbors[i] = []
            
            # Group neighbors by instance
            for i, j in zip(rows, cols):
                if i != j:  # Skip self
                    instance_neighbors[i].append(j)
            
            # Calculate the mean of the target values for each instance's neighbors
            nn_y_avg = np.zeros(len(df_))
            for i in range(len(df_)):
                if instance_neighbors[i]:  # If there are neighbors
                    nn_y_avg[i] = np.mean(regression_measures.y_scaled[instance_neighbors[i]])
                else:  # If no neighbors (shouldn't happen with n_neighbors+1)
                    nn_y_avg[i] = regression_measures.y_scaled[i]
            
            # Calculate the squared error
            S3 = (nn_y_avg - regression_measures.y_scaled) ** 2
            
            # Normalize and return
            return 1 - np.exp(-S3 / S3.std())
        except Exception as e:
            logger.error(f"Error in alternative_error_nn_regressor: {str(e)}")
            # Return a default value for all instances
            return np.ones(len(df_)) * 0.5
    
    # Replace the method with our fixed version
    # regression_measures.error_nn_regressor = alternative_error_nn_regressor
    
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