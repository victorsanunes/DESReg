from desReg.des.DESRegression import DESRegression

import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

#Three kind of ensembles (DES, DRE and SE) are compared. With this aim, 5 fold cross validation and nine datasets are used.


ensemble = DESRegression(regressors_list = [linear_model.Lasso(), linear_model.Ridge(), KNeighborsRegressor(), 
                                             SVR(), DecisionTreeRegressor()], n_estimators_bag=20, random_state=0)

#partition_name = './Datasets/Abalone/abalone-5-'
#partition_name = './Datasets/Concrete/concrete-5-'
#partition_name = './Datasets/Liver/liver-5-'
#partition_name = './Datasets/Machine_CPU/machineCPU-5-'
#partition_name = './Datasets/Real_estate/Real_estate-5-'
partition_name = './Datasets/Student_marks/student_marks-5-' 
#partition_name = './Datasets/Wine_quality_red/winequality-red-5-'
#partition_name = './Datasets/Wine_quality_white/winequality-white-5-'
#partition_name = './Datasets/Yacht/yacht_hydrodynamics-5-'

n_fold = 5

errors_DES = []
errors_DRS = []
errors_SE = []
for p in range(1,n_fold+1):
    name_train = partition_name+str(p)+'tra.dat'
    print(name_train)
    name_test = partition_name+str(p)+'tst.dat'
    print(name_test)
    
    data_train = pd.read_csv(name_train,header = None)
    X_train = data_train.iloc[:,:-1].to_numpy()
    y_train = np.ravel(data_train.iloc[:, -1:])

    data_test = pd.read_csv(name_test, header=None)
    X_test = data_test.iloc[:,:-1].to_numpy()
    y_test = np.ravel(data_test.iloc[:, -1:])

    
    ensemble.fit(X_train, y_train)

        
    print('Partition '+ str(p))
    
    params = {'ensemble_type': 'DES'} 
    y_pred_DES = ensemble.predict(X_test, params)
    error = mean_squared_error(y_test, y_pred_DES)
    errors_DES.append(error)
    print('DES error prediction:', error)
    
    params = {'ensemble_type': 'DRS'}
    y_pred_DRS = ensemble.predict(X_test, params)
    error = mean_squared_error(y_test, y_pred_DRS)
    errors_DRS.append(error)
    print('DRS error prediction:', error)
    
    params = {'ensemble_type': 'SE'}
    y_pred_SE = ensemble.predict(X_test, params)
    error = mean_squared_error(y_test, y_pred_SE)
    errors_SE.append(error)
    print('SE error prediction:', error)

mean_errors_DES = np.mean(errors_DES)
print('Mean error DES', mean_errors_DES)

mean_errors_DRS = np.mean(errors_DRS)
print('Mean error DRS', mean_errors_DRS)

mean_errors_SE = np.mean(errors_SE)
print('Mean error SE', mean_errors_SE)






