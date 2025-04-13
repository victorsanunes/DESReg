from desReg.des.DESRegression import DESRegression
from desReg.dataset import load_Student_Mark

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from scipy.spatial import distance
import pandas as pd

# Define a custom distance function for output profiles
def output_profile_distance(v1, v2):
    # Ensure both vectors are numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # If v1 is a single value (from DSEL), convert it to a 1D array
    if np.isscalar(v1):
        v1 = np.array([v1])
    
    # If v2 is a list of predictions from multiple regressors
    if len(v2.shape) > 1:
        v2 = v2.flatten()
    
    # Calculate Euclidean distance between the vectors
    return distance.euclidean(v1, v2)

data = load_Student_Mark()
X = data.iloc[:,:-1].to_numpy()
y = np.ravel(data.iloc[:, -1:]) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

partitions_dict = {
    'student_marks': './Datasets/Student_marks/student_marks-5-', # Funcionou
}

# Create empty DataFrame to store results
results = pd.DataFrame(columns=['Dataset', 'Method', 'Fold', 'MSE'])

def run_experiment(dataset_alias, n_fold=5):
     partition_name = partitions_dict[dataset_alias]
     heterogeneous_DES = DESRegression(
          regressors_list = [Lasso(alpha = 0.15),SVR()], 
          n_estimators_bag = 10, 
          DSEL_perc = 0.95, 
          XTRAIN_full = True, 
          distance = output_profile_distance, 
          competence_region = 'output_profiles', 
          competence_level= mean_squared_error
     )

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
          heterogeneous_DES.fit(X_train, y_train)

          # Get predictions and errors
          y_pred_des = heterogeneous_DES.predict(X_test)
          
          params = {'ensemble_type': 'DRS'}
          y_pred_DRS = heterogeneous_DES.predict(X_test, params)
          
          # The ensemble type is changed to DE, all the
          # regressors in pool are chosen for prediction
          params = {'ensemble_type': 'SE'}
          y_pred_SE = heterogeneous_DES.predict(X_test, params)

          # Calculate errors
          mse_des = mean_squared_error(y_test, y_pred_des)
          mse_DRS = mean_squared_error(y_test, y_pred_DRS)
          mse_SE = mean_squared_error(y_test, y_pred_SE)


          # Add results to DataFrame
          results.loc[len(results)] = [dataset_alias, 'DES', fold, mse_des]
          results.loc[len(results)] = [dataset_alias, 'DRS', fold, mse_DRS]
          results.loc[len(results)] = [dataset_alias, 'SE', fold, mse_SE]
          print(results)


run_experiment('student_marks')

print("\n======Agg results======")
agg_results = results.groupby(['Method', 'Dataset']).agg({'MSE': 'mean'}).reset_index()
print(agg_results)