import pandas as pd
from pyhard.measures import RegressionMeasures
from pyhard.regression import RegressorsPool

def build_pyhard_df(X, y):
    df = pd.DataFrame(X)
    df['y'] = y
    return df

def consolidate_data(
    df, 
    regression_measures_df, 
    performances_df, 
    instance_hardness):
    
    all_consolidated_df = pd.concat([
        df.reset_index().drop(columns=['index']),
        regression_measures_df,
        performances_df,
        instance_hardness
    ],
    axis=1
    )
    return all_consolidated_df

def compute_regression_measures(df: dict) -> tuple:
    # _measures_dict = {
    #     'C4': 'collective_feature_efficiency',
    #     'L1': 'linear_absolute_error',
    #     'S1': 'output_distribution',
    #     'S2': 'input_distribution',
    #     'S3': 'error_nn_regressor',
    #     'FO': 'feature_outlier',
    #     'POT': 'target_outlier',
    #     'HB': 'histogram_bin',
    #     'EDS': 'stump_error',
    #     'EZR': 'zero_rule_error',
    #     'DS': 'disjunct_size',
    #     'TD_P': 'tree_depth_pruned',
    #     'TD_U': 'tree_depth_unpruned',
    #     'D': 'density',
    #     'CC': 'clustering_coefficient'
    # }
    df_ = df.copy()
    regression_measures = RegressionMeasures(df_)
    regression_measures_df = regression_measures.calculate_all(measures_list=['L1', 'S1', 'S2', 'CC'])
    # regression_measures_df = regression_measures.calculate_all()
    return regression_measures_df

def create_pool_regressors(df: pd.DataFrame) -> RegressorsPool:
    pool_regressors = RegressorsPool(df)
    return pool_regressors

def compute_algorithm_performance(pool_regressors):
    performances_df = pool_regressors.run_all(
        hyper_param_optm=False, 
        n_folds=5, 
        n_iter=1)
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
    # return instance_hardness_df