import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='The default value of `dual` will change')

from desReg.des.DESRegression import DESRegression
from desReg.utils import measures as em

import numpy as np
import pandas as pd
import datetime as dt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

#Three kind of ensembles (DES, DRE and SE) are compared. With this aim, 5 fold cross validation and nine datasets are used.

partitions_dict = {
    # 'abalone': './Datasets/Abalone/abalone-5-',
    # 'concrete': './Datasets/Concrete/concrete-5-',
    # 'liver': './Datasets/Liver/liver-5-',
    # 'machineCPU': './Datasets/MachineCPU/machineCPU-5-',
    # 'real_estate': './Datasets/Real_estate/Real_estate-5-',
    'student_marks': './Datasets/Student_marks/student_marks-5-',
    'wine_quality_red': './Datasets/Wine_quality_red/winequality-red-5-',
    'wine_quality_white': './Datasets/Wine_quality_white/winequality-white-5-',
    'yacht': './Datasets/Yacht/yacht_hydrodynamics-5-'
}

# Create empty DataFrame to store results
results = pd.DataFrame(columns=['Dataset', 'Method', 'Fold', 'MSE', 'R2', 'MAE', 'RMSE', 'MAPE'])
regressors_list = [
    linear_model.Lasso(), 
    linear_model.Ridge(), 
    KNeighborsRegressor(), 
    SVR(), DecisionTreeRegressor()
]

def run_experiment(dataset_alias, n_fold=5):
    partition_name = partitions_dict[dataset_alias]
    ensemble = DESRegression(
        regressors_list=regressors_list,
        random_state=0,
        n_estimators_bag=20,
        competence_region='cluster',
        k=3,
        competence_level=em.sum_absolute_error,
        include_instance_hardness=False
    )

    ensemble_ih = DESRegression(
        regressors_list=regressors_list,
        random_state=0,
        n_estimators_bag=20,
        competence_region='cluster',
        k=5,
        competence_level=em.sum_absolute_error_weighted,
        include_instance_hardness=True
    )

    linear_reg = linear_model.LinearRegression()

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

        # Fit models
        ensemble.fit(X_train, y_train)
        ensemble_ih.fit(X_train, y_train)
        linear_reg.fit(X_train, y_train)

        # Get predictions and errors
        y_pred_des = ensemble.predict(X_test, {'ensemble_type': 'SE'})
        y_pred_des_ih = ensemble_ih.predict(X_test, {'ensemble_type': 'DES'})
        y_pred_linear = linear_reg.predict(X_test)

        # Calculate errors
        mse_des = mean_squared_error(y_test, y_pred_des)
        r2_des = r2_score(y_test, y_pred_des)
        mae_des = mean_absolute_error(y_test, y_pred_des)
        rmse_des = np.sqrt(mse_des)
        mape_des = mean_absolute_percentage_error(y_test, y_pred_des)
        
        # Ensemble with instance hardness
        mse_des_ih = mean_squared_error(y_test, y_pred_des_ih)
        r2_des_ih = r2_score(y_test, y_pred_des_ih)
        mae_des_ih = mean_absolute_error(y_test, y_pred_des_ih)
        rmse_des_ih = np.sqrt(mse_des_ih)
        mape_des_ih = mean_absolute_percentage_error(y_test, y_pred_des_ih)
        
        # Linear regression
        mse_linear = mean_squared_error(y_test, y_pred_linear)
        r2_linear = r2_score(y_test, y_pred_linear)
        mae_linear = mean_absolute_error(y_test, y_pred_linear)
        rmse_linear = np.sqrt(mse_linear)
        mape_linear = mean_absolute_percentage_error(y_test, y_pred_linear)

        # Add results to DataFrame
        results.loc[len(results)] = [dataset_alias, 'DES', fold, mse_des, r2_des, mae_des, rmse_des, mape_des]
        results.loc[len(results)] = [dataset_alias, 'DES_IH', fold, mse_des_ih, r2_des_ih, mae_des_ih, rmse_des_ih, mape_des_ih]
        results.loc[len(results)] = [dataset_alias, 'Linear', fold, mse_linear, r2_linear, mae_linear, rmse_linear, mape_linear]

# Run experiments for all datasets
for dataset in partitions_dict.keys():
    try:
        run_experiment(dataset)
    except Exception as e:
        print(f"Error processing {dataset}: {str(e)}")

# Calculate summary statistics
summary = results.groupby(['Dataset', 'Method']).agg({
    'MSE': ['mean', 'std'],
    'R2': ['mean', 'std'],
    'MAE': ['mean', 'std'],
    'RMSE': ['mean', 'std'],
    'MAPE': ['mean', 'std']
}).round(4)

# Save results
timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
results.to_csv(f'detailed_results_{timestamp}.csv', index=False)
summary.to_csv(f'summary_results_{timestamp}.csv')

# Print summary table
print("\n=== Summary Results ===")
print(summary)