import warnings

from desReg.des.DESRegression import DESRegression
from desReg.utils import measures as em

import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
from scipy.spatial import distance
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='The default value of `dual` will change')

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
            import logging
            from inspect import signature
            
            # Configure logging
            logging.basicConfig(level=logging.INFO, 
                               format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            logger = logging.getLogger(f"model_{model_name}")
            
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
                            return model.predict(X_test, params=params)
                        else:
                            logger.info("Predict method doesn't accept params or params is None")
                            return model.predict(X_test)
                    
                    # Submit the function to the executor
                    future = executor.submit(fit_and_predict)
                    
                    # Wait for the result with timeout
                    y_pred = future.result(timeout=timeout_seconds)
                    
                # Calculate execution time
                execution_time = time.time() - start_time
                logger.info(f"{model_name} model execution completed in {execution_time:.2f} seconds")
                
                # Calculate errors
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                
                logger.info(f"{model_name} model metrics - MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")
                
                return {
                    'mse': mse,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'execution_time': execution_time,
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
            # logger.info(f"Challenger model {model_name} results (10 first results): {challenger_result}")
        models_results[fold] = fold_results
    
    return models_results

def get_df_results(model_results):
    results_df = pd.DataFrame()
    for partition, partition_results in model_results.items():
        # Iterate through each model in the dictionary
        for model_name, metrics in partition_results.items():
            # Create a row with all metrics
            row = metrics.copy()
            # Add the model name
            row['model_name'] = model_name
            row['partition'] = partition
            # Append the row to the DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    return results_df

def get_agg_results(results_df):
    agg_results = results_df.groupby(['model_name']).agg({
        'mse': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'mape': ['mean', 'std'],
        'execution_time': ['mean', 'std', 'sum']
    }).round(4)
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
        hardness_measures_list=['S3'],
        meta_feature='S3'
    )
    
    logger.info("Running experiments")
    results = run_experiment_with_timeout(
        'yacht', 
        baseline_model=linear_model.Lasso(), 
        challenger_experiments = {
            'ensemble_hm_s3': {
            "model": ensemble_hm,
            "params": {
                "include_hardness_measures": True,
                "include_instance_hardness": False,
                "meta_feature": "S3",
                    "hardness_measures_list": ["S3"]
                }
            }
        }
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