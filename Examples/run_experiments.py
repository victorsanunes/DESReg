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

def run_experiment_with_timeout(
    dataset_alias, 
    n_fold=5,
    baseline_model = None,
    challenger_experiments: dict = None,
    timeout_seconds=30*60,  # Default timeout of 5 minute
):
    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    
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
        
        # Function to run model with timeout
        def run_model_with_timeout(
            model, 
            model_name, 
            X_train, y_train, X_test, y_test, 
            timeout_seconds, params=None
        ):
            import traceback
            from inspect import signature
            
            start_time = time.time()
            logger.info(f"Starting {model_name} model execution")
            
            try:
                # Use ThreadPoolExecutor to run the model with a timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    logger.info(f"Fitting {model_name} model")
                    
                    # Define a function that handles the predict method with or without params
                    def fit_and_predict():
                        # Fit the model
                        model.fit(X_train, y_train)
                        
                        # Check if predict method accepts params
                        predict_sig = signature(model.predict)
                        if 'params' in predict_sig.parameters and params is not None:
                            logger.info(f"Using params in predict: {params}")
                            return model.predict(X_test, params=params, return_ensemble_sizes=True)
                        else:
                            logger.info("Predict method doesn't accept params or params is None")
                            return model.predict(X_test, return_ensemble_sizes=True)
                    
                    # Submit the function to the executor
                    future = executor.submit(fit_and_predict)
                    
                    # Wait for the result with timeout
                    y_pred = future.result(timeout=timeout_seconds)
                    
                # Calculate execution time
                execution_time = time.time() - start_time
                logger.info(f"{model_name} model execution completed in {execution_time:.2f} seconds")
                
                # Calculate errors
                y_pred, ensemble_sizes = y_pred
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                total = sum(ensemble_sizes, Counter())
                logger.info(f"{model_name} model metrics - MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")
                
                return {
                    'mse': mse,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'execution_time': execution_time,
                    'y_pred': y_pred,
                    'y_true': y_test,
                    'ensemble_sizes': total,
                    'status': 'success'
                }
            except TimeoutError:
                execution_time = time.time() - start_time
                logger.error(f"{model_name} model execution timed out after {execution_time:.2f} seconds")
                return {
                    'mse': None,
                    'r2': None,
                    'mae': None,
                    'rmse': None,
                    'mape': None,
                    'execution_time': execution_time,
                    'y_true': y_test,
                    'y_pred': None,
                    'ensemble_sizes': None,
                    'status': f'timeout after {execution_time:.2f} seconds'
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
        
        # Run baseline model
        print(f"Running baseline model...")
        baseline_result = run_model_with_timeout(
            baseline_model, 'baseline', X_train, y_train, X_test, y_test, timeout_seconds, params=None
        )
        fold_results['baseline'] = baseline_result
        
        # Run challenger models
        for model_name in challenger_experiments.keys():
            print(f"Running challenger model: {model_name}...")
            model = challenger_experiments[model_name]['model']   
            params = challenger_experiments[model_name].get("params", None)
            challenger_result = run_model_with_timeout(
                model, model_name, X_train, y_train, X_test, y_test, timeout_seconds, params
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

def sum_freqs(freqs):
    return sum(freqs, Counter())

def get_agg_results(results_df):
    agg_results = results_df.groupby(['model_name']).agg({
        'mse': ['mean', 'std'],
        # 'r2': ['mean', 'std'],
        # 'mae': ['mean', 'std'],
        # 'rmse': ['mean', 'std'],
        # 'mape': ['mean', 'std'],
        'ensemble_sizes': [sum_freqs],
        'execution_time': ['mean', 'std', 'sum'],
        'partition_size': ['sum']
    }).round(4)
    ensembles_counter_df = pd.json_normalize(agg_results[('ensemble_sizes', 'sum_freqs')])\
        .set_index(agg_results.index)
    ensembles_counter_df.columns = pd.MultiIndex.from_product([['algo_name'], ensembles_counter_df.columns])

    df_concat = pd.concat([agg_results, ensembles_counter_df], axis=1)
    return df_concat

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
    results = run_experiment_with_timeout(
        'yacht', 
        baseline_model=ensemble, 
        challenger_experiments = challenger_experiments
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
    baseline_model = None,
    challenger_experiments: dict = None,
):
    meta_data = external_datasets[dataset_name]()
    dataset = meta_data['data']
    X = dataset[meta_data['features']]
    y = dataset[meta_data['target_column']]
    
    # Dicti with fold results. Each key is a fold and each value is a dictionary with the results of the model
    fold_results = {}
    # Run 5-fold cross validation
    partitions = 5
    kf = KFold(n_splits=partitions, shuffle=True, random_state=0)

    partition_number = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        for model_name in challenger_experiments.keys():
            # initialize time
            start_time = time.time()
            model = challenger_experiments[model_name]['model']
            params = challenger_experiments[model_name].get("params", None)
            # Fit the baseline model
            model.fit(X_train, y_train)
            execution_time = time.time() - start_time

            # Predict the test set
            y_pred = model.predict(X_test, params=params)

            # Calculate the metrics
            mse = mean_squared_error(y_test, y_pred)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            # Calculate execution time

            # Add the results to the fold results
            fold_results[model_name] = {
                'mse': mse,
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'execution_time': execution_time,
                'y_pred': y_pred,
                'y_true': y_test,
                'status': 'success'
            }