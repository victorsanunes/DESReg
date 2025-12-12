import warnings

from desReg.des.DESRegression import DESRegression
from desReg.utils import measures as em
from desReg.dataset.load_datasets import load_air_quality, load_airfoil, load_ccpp

import numpy as np
import pandas as pd
import logging
import time
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.base import clone
from collections import Counter
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='The default value of `dual` will change')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

partitions_dict = {
    'abalone': './Datasets/Abalone/abalone-5-', # demora pra rodar
    'concrete': './Datasets/Concrete/concrete-5-', #nao funcionou
    'liver': './Datasets/Liver/liver-5-', #Funcionou
    'machineCPU': './Datasets/MachineCPU/machineCPU-5-', # Funcionou
    'real_estate': './Datasets/Real_estate/Real_estate-5-', # Funcionou
    'student_marks': './Datasets/Student_marks/student_marks-5-', # Funcionou
    'wine_quality_red': './Datasets/Wine_quality_red/winequality-red-5-', # -> demora pra rodar
    'wine_quality_white': './Datasets/Wine_quality_white/winequality-white-5-', #-> demora pra rodar
    'yacht': './Datasets/Yacht/yacht_hydrodynamics-5-' # Funcionou
}

def calculate_metrics(y_true, y_pred, ensemble_sizes=None):
    """
    Calculate regression metrics for predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    ensemble_sizes : list of Counter objects, optional
        Ensemble sizes for DES models
        
    Returns:
    --------
    dict : Dictionary containing all calculated metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    metrics = {
        'mse': mse,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }
    
    if ensemble_sizes is not None:
        total = sum(ensemble_sizes, Counter())
        # Calculate average by dividing each count by the number of samples
        n_samples = len(ensemble_sizes)
        average = {key: count / n_samples for key, count in total.items()}
        metrics['ensemble_sizes'] = average
    else:
        metrics['ensemble_sizes'] = None
    
    return metrics


def run_model(model, model_name, X_train, y_train, X_test, y_test, params=None):
    """
    Train and evaluate a regression model.
    
    Parameters:
    -----------
    model : estimator object
        The model to train and evaluate
    model_name : str
        Name of the model for logging purposes
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like
        Test data
    params : dict, optional
        Additional parameters to pass to the predict method
        
    Returns:
    --------
    dict : Dictionary containing metrics, predictions, and execution info
    """
    import traceback
    from inspect import signature
    
    start_time = time.time()
    logger.info(f"Starting {model_name} model execution")
    
    try:
        # Fit the model
        logger.info(f"Fitting {model_name} model")
        model.fit(X_train, y_train)
        
        # Check if predict method accepts params and return_ensemble_sizes
        predict_sig = signature(model.predict)
        
        # Determine how to call predict based on its signature
        if 'return_ensemble_sizes' in predict_sig.parameters:
            # This is a DESRegression model
            if 'params' in predict_sig.parameters and params is not None:
                logger.info(f"Using params in predict: {params}")
                y_pred, ensemble_sizes = model.predict(X_test, params=params, return_ensemble_sizes=True)
            else:
                logger.info("Predict method accepts return_ensemble_sizes but no params")
                y_pred, ensemble_sizes = model.predict(X_test, return_ensemble_sizes=True)
        else:
            # This is a standard sklearn model
            logger.info("Standard model - predict without ensemble_sizes")
            y_pred = model.predict(X_test)
            ensemble_sizes = None
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"{model_name} model execution completed in {execution_time:.2f} seconds")
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, ensemble_sizes)
        logger.info(f"{model_name} model metrics - MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}, "
                   f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, MAPE: {metrics['mape']:.4f}")
        
        return {
            **metrics,
            'execution_time': execution_time,
            'y_pred': y_pred,
            'y_true': y_test,
            'status': 'success'
        }
    except Exception as e:
        execution_time = time.time() - start_time
        error_traceback = traceback.format_exc()
        logger.error(f"{model_name} model execution failed with error: {str(e)}")
        logger.error(f"Error traceback: {error_traceback}")
        
        # Log model information for debugging
        logger.error(f"Model type: {type(model).__name__}")
        logger.error(f"Model parameters: {getattr(model, 'get_params', lambda: {})()}")
        
        return {
            'mse': None,
            'r2': None,
            'mae': None,
            'rmse': None,
            'mape': None,
            'execution_time': execution_time,
            'y_pred': None,
            'y_true': y_test,
            'ensemble_sizes': None,
            'status': f'error: {str(e)}',
            'error_traceback': error_traceback
        }


def run_experiment(
    dataset_alias, 
    n_fold=5,
    baseline_model=None,
    challenger_experiments: dict = None,
):
    """
    Run experiments on a dataset with baseline and challenger models.
    
    Parameters:
    -----------
    dataset_alias : str
        Alias for the dataset in partitions_dict
    n_fold : int
        Number of folds to process
    baseline_model : estimator object, optional
        Baseline model to compare against
    challenger_experiments : dict
        Dictionary of challenger models with their parameters
        
    Returns:
    --------
    dict : Dictionary containing results for each fold
    """
    partition_name = partitions_dict[dataset_alias]
    models_results = {}
    
    for fold in range(1, n_fold + 1):
        print(f'========== Processing {dataset_alias} - Fold {fold} ==========')
        
        # Load data
        name_train = partition_name + str(fold) + 'tra.dat'
        name_test = partition_name + str(fold) + 'tst.dat'
        
        data_train = pd.read_csv(name_train, header=None)
        X_train = data_train.iloc[:, :-1].to_numpy()
        y_train = np.ravel(data_train.iloc[:, -1:])

        data_test = pd.read_csv(name_test, header=None)
        X_test = data_test.iloc[:, :-1].to_numpy()
        y_test = np.ravel(data_test.iloc[:, -1:])

        # Initialize fold results
        fold_results = {}
        
        # Run baseline model if provided
        if baseline_model is not None:
            print("Running baseline model...")
            baseline_result = run_model(
                baseline_model, 'baseline', X_train, y_train, X_test, y_test, params=None
            )
            fold_results['baseline'] = baseline_result
        
        # Run challenger models
        for model_name in challenger_experiments.keys():
            print(f"Running challenger model: {model_name}...")
            model_template = challenger_experiments[model_name]['model']   
            model = clone(model_template)
            params = challenger_experiments[model_name].get("params", None)
            challenger_result = run_model(
                model, model_name, X_train, y_train, X_test, y_test, params
            )
            fold_results[model_name] = challenger_result
        
        models_results[fold] = fold_results
    
    return models_results

def get_df_results(model_results):
    results_df = pd.DataFrame()
    for partition, partition_results in model_results.items():
        # Iterate through each model in the dictionary
        for model_name, metrics in partition_results.items():
            # Create a row with all metrics
            row = metrics.copy()    
            if isinstance(row['y_pred'][0], np.ndarray):
                row['y_pred'] = np.concatenate(row['y_pred']).tolist()
            # Add the model name
            row['model_name'] = model_name
            row['partition'] = partition
            row['partition_size'] = len(row['y_pred'])
            # Append the row to the DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    return results_df

def average_ensemble_sizes(ensemble_sizes_list):
    """
    Calculate the mean of ensemble sizes across folds.
    Each element in the list is a dict with algorithm names as keys and their average usage as values.
    """
    if not ensemble_sizes_list or all(es is None for es in ensemble_sizes_list):
        return None
    
    # Filter out None values
    valid_sizes = [es for es in ensemble_sizes_list if es is not None]
    if not valid_sizes:
        return None
    
    # Sum all dictionaries
    total = {}
    for es in valid_sizes:
        for key, value in es.items():
            total[key] = total.get(key, 0) + value
    
    # Calculate average
    n_folds = len(valid_sizes)
    average = {key: count / n_folds for key, count in total.items()}
    return average

def get_agg_results(results_df):
    # Expand ensemble_sizes dictionary into separate columns
    ensemble_sizes_expanded = pd.json_normalize(results_df['ensemble_sizes'])
    
    # Add prefix to the expanded columns to identify them
    ensemble_sizes_expanded = ensemble_sizes_expanded.add_prefix('ensemble_size_')
    
    # Concatenate with original dataframe (excluding the original ensemble_sizes column)
    results_df_expanded = pd.concat([
        results_df.drop('ensemble_sizes', axis=1), 
        ensemble_sizes_expanded
    ], axis=1)
    
    # Create aggregation dictionary dynamically for ensemble_size columns
    agg_dict = {
        'mse': ['mean', 'std'],
        # 'r2': ['mean', 'std'],
        # 'mae': ['mean', 'std'],
        # 'rmse': ['mean', 'std'],
        # 'mape': ['mean', 'std'],
        'execution_time': ['mean', 'std', 'sum'],
        'partition_size': ['sum']
    }
    
    # Add all ensemble_size columns to aggregation
    for col in ensemble_sizes_expanded.columns:
        agg_dict[col] = ['mean']
    
    # Perform groupby aggregation
    agg_results = results_df_expanded.groupby(['model_name']).agg(agg_dict).round(4)
    
    return agg_results

if __name__ == "__main__":
    import logging
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("experiment_results.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("experiment")
    
    regressors_list = [
        linear_model.Lasso(), 
        linear_model.Ridge(), 
        KNeighborsRegressor(), 
        SVR(), 
        DecisionTreeRegressor()
    ]
    
    logger.info("Initializing ensemble models")
    ensemble = DESRegression(
        regressors_list=regressors_list,
        random_state=0,
        n_estimators_bag=20,
        competence_region='knn',
        k=5,
        competence_level=em.all_errors,
        include_instance_hardness=False
    )
    ensemble_ih = DESRegression(
        regressors_list=regressors_list,
        random_state=0,
        n_estimators_bag=20,
        competence_region='knn',
        k=5,
        competence_level=em.sum_absolute_error_weighted,
        include_instance_hardness=True,
        include_hardness_measures=False,
        hardness_measures_list=None,
        meta_feature='instance_hardness'
    )

    ensemble_hm = DESRegression(
        regressors_list=regressors_list,
        random_state=0,
        n_estimators_bag=20,
        competence_region='knn',
        k=5,
        competence_level=em.sum_absolute_error_weighted,
        include_instance_hardness=False,
        include_hardness_measures=True,
        hardness_measures_list=['S1','S2','S3', 'D', 'L1', 'EZR', 'EDS'],
        meta_feature='S3'
    )
    linear_reg = linear_model.LinearRegression()
    challenger_experiments = {
        'ensemble_ih': {
            "model": ensemble_ih,
            "params": {
                "include_hardness_measures": False,
                "include_instance_hardness": True,
                "meta_feature": "instance_hardness"
            }
        },
        'ensemble_hm_s2': {
            "model": ensemble_hm,
            "params": {
                "include_hardness_measures": True,
                "include_instance_hardness": False,
                "meta_feature": "S2",
                # "hardness_measures_list": ["S2"]
            }
        },
        'ensemble_hm_s1': {
            "model": ensemble_hm,
            "params": {
                "include_hardness_measures": True,
                "include_instance_hardness": False,
                "meta_feature": "S1",
                # "hardness_measures_list": ["S1"]
            }
        },
        'ensemble_hm_s3': {
            "model": ensemble_hm,
            "params": {
                "include_hardness_measures": True,
                "include_instance_hardness": False,
                "meta_feature": "S3",
                # "hardness_measures_list": ["S3"]
            }
        },
        'ensemble_hm_d': {
            "model": ensemble_hm,
            "params": {
                "include_hardness_measures": True,
                "include_instance_hardness": False,
                "meta_feature": "D",
                # "hardness_measures_list": ["D"]
            }
        },
        'ensemble_hm_l1': {
            "model": ensemble_hm,
            "params": {
                "include_hardness_measures": True,
                "include_instance_hardness": False,
                "meta_feature": "L1",
                # "hardness_measures_list": ["L1"]
            }
        },
        'ensemble_hm_ezr': {
            "model": ensemble_hm,
            "params": {
                "include_hardness_measures": True,
                "include_instance_hardness": False,
                "meta_feature": "EZR",
                # "hardness_measures_list": ["EZR"]
            }
        },
        'ensemble_hm_eds': {
            "model": ensemble_hm,
            "params": {
                "include_hardness_measures": True,
                "include_instance_hardness": False,
                "meta_feature": "EDS",
                # "hardness_measures_list": ["EDS"]
            }
        },
        'linear_reg': {
            "model": linear_reg,
        }
    }
    
    logger.info("Running experiments")
    results = run_experiment(
        'yacht', 
        baseline_model=ensemble, 
        challenger_experiments=challenger_experiments
    )

    # Process and display results
    logger.info("Processing results")
    results_df = get_df_results(results)

    # Check for errors
    error_models = results_df[results_df['status'].str.startswith('error')]
    if not error_models.empty:
        logger.error(f"Found {len(error_models)} models with errors:")
        for _, row in error_models.iterrows():
            logger.error(f"Model: {row['model_name']}, Dataset: {row['partition']}, Error: {row['status']}")
            if 'error_traceback' in row and row['error_traceback']:
                logger.error(f"Traceback: {row['error_traceback']}")
    
    # Display summary statistics
    logger.info("Summary statistics:")
    summary = results_df.groupby('model_name').agg({
        'mse': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'mape': ['mean', 'std'],
        'execution_time': ['mean', 'std', 'sum']
    }).round(4)
    
    print("\nResults Summary:")
    print(summary)
    
    # Save results to CSV
    # results_df.to_csv("experiment_results.csv", index=False)
    logger.info("Results saved to experiment_results.csv")


external_datasets = {
    'air_quality': load_air_quality,
    'airfoil': load_airfoil,
    'ccpp': load_ccpp,
}
def run_experiment_with_external_datasets(
    dataset_name,
    baseline_model=None,
    challenger_experiments: dict = None,
    n_splits=5,
    random_state=0,
):
    """
    Run experiments on external datasets using k-fold cross-validation.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the external dataset to load
    baseline_model : estimator object, optional
        Baseline model to compare against
    challenger_experiments : dict
        Dictionary of challenger models with their parameters
    n_splits : int
        Number of folds for cross-validation
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    dict : Dictionary containing results for each fold
    """
    meta_data = external_datasets[dataset_name]()
    dataset = meta_data['data']
    X = dataset[meta_data['features']]
    y = dataset[meta_data['target_column']]
    
    # Dictionary with fold results. Each key is a fold number and each value is a dictionary with model results
    models_results = {}
    
    # Run k-fold cross validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_number, (train_index, test_index) in enumerate(kf.split(X), start=1):
        print(f'========== Processing {dataset_name} - Fold {fold_number} ==========')
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        fold_results = {}
        
        # Run baseline model if provided
        if baseline_model is not None:
            print("Running baseline model...")
            baseline_result = run_model(
                baseline_model, 'baseline', X_train, y_train, X_test, y_test, params=None
            )
            fold_results['baseline'] = baseline_result

        # Run challenger models
        for model_name in challenger_experiments.keys():
            print(f"Running challenger model: {model_name}...")
            model_template = challenger_experiments[model_name]['model']
            model = clone(model_template)
            params = challenger_experiments[model_name].get("params", None)
            
            challenger_result = run_model(
                model, model_name, X_train, y_train, X_test, y_test, params
            )
            fold_results[model_name] = challenger_result
        
        models_results[fold_number] = fold_results
    
    return models_results