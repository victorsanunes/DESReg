from desReg.des.DESRegression import DESRegression
from desReg.utils import measures as em

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

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

#Function to load data based on the dataset name
def load_data(dataset_name):
    partition_name = partitions_dict[dataset_name]

    n_fold = 5
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for p in range(1, n_fold+1):
        name_train = partition_name+str(p)+'tra.dat'
        name_test = partition_name+str(p)+'tst.dat'

        data_train = pd.read_csv(name_train,header = None)
        columns = [f"V{i}" for i in range(data_train.shape[1] - 1)]
        columns.append("y")
        data_train.columns = columns
        X_train = data_train.iloc[:,:-1].to_numpy()
        y_train = np.ravel(data_train.iloc[:, -1:])

        data_test = pd.read_csv(name_test, header=None)
        data_test.columns = columns
        X_test = data_test.iloc[:,:-1].to_numpy()
        y_test = np.ravel(data_test.iloc[:, -1:])

        train_df = pd.concat([train_df, data_train], ignore_index=True)
        test_df = pd.concat([test_df, data_test], ignore_index=True)
    
    dataset_object = dict(
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        train_df = train_df,
        test_df = test_df
    )
    return dataset_object
def get_sampled_df(dataset_object, n_samples_train, n_samples_test):
    for data in dataset_object:
        sampled_train_df = dataset_object['train_df'].sample(n=n_samples_train, random_state=42)
        sampled_test_df = dataset_object['test_df'].sample(n=n_samples_test, random_state=42)

        sampled_X_train = sampled_train_df.iloc[:,:-1].to_numpy()
        sampled_y_train = np.ravel(sampled_train_df.iloc[:, -1:])
        sampled_X_test = sampled_test_df.iloc[:,:-1].to_numpy()
        sampled_y_test = np.ravel(sampled_test_df.iloc[:, -1:])

        sampled_dataset_object = dict(
            X_train = sampled_X_train,
            y_train = sampled_y_train,
            X_test = sampled_X_test,
            y_test = sampled_y_test,
            train_df = sampled_train_df,
            test_df = sampled_test_df
        )
    return sampled_dataset_object

# ensemble_hm_v1 = DESRegression(
#     regressors_list = [Lasso(alpha = 0.15), SVR()],
#     random_state=42,
#     n_estimators_bag=10,
#     competence_region='cluster',
#     k=3,
#     competence_level=em.sum_absolute_error_weighted,
#     include_instance_hardness=False,
#     include_hardness_measures=True,
#     hardness_measures_list=['S2'],
#     meta_feature='S2'
# )

# ensemble_hm_v2 = DESRegression(
#     regressors_list = [Lasso(alpha = 0.15), SVR()],
#     random_state=42,
#     n_estimators_bag=10,
#     competence_region='knn',
#     k=3,
#     competence_level=em.sum_absolute_error_weighted,
#     include_instance_hardness=False,
#     include_hardness_measures=True,
#     hardness_measures_list=['S2'],
#     meta_feature='S2'
# )

regressors_list = [
    linear_model.Lasso(), 
    linear_model.Ridge(), 
    KNeighborsRegressor(), 
    SVR(), 
    DecisionTreeRegressor()
]

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
    hardness_measures_list=['L1'],
    meta_feature='L1'
)

dataset_object = load_data('yacht')
# sampled_dataset_object = get_sampled_df(dataset_object, 500, 100)
sampled_dataset_object = dataset_object.copy()
ensemble_ih.fit(sampled_dataset_object['X_train'], sampled_dataset_object['y_train'])
ensemble_hm.fit(sampled_dataset_object['X_train'], sampled_dataset_object['y_train'])

y_pred_v1 = ensemble_ih.predict(sampled_dataset_object['X_test'])
breakpoint()
y_pred_v2 = ensemble_hm.predict(sampled_dataset_object['X_test'])
breakpoint()

print('MSE error v1:', mean_squared_error(sampled_dataset_object['y_test'], y_pred_v1))
print('MSE error v2:', mean_squared_error(sampled_dataset_object['y_test'], y_pred_v2))
breakpoint()