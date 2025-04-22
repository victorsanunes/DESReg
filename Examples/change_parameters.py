from desReg.des.DESRegression import DESRegression
from desReg.dataset import load_Student_Mark

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Lasso

from scipy.spatial import distance

data = load_Student_Mark()
X = data.iloc[:,:-1].to_numpy()
y = np.ravel(data.iloc[:, -1:]) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

heterogeneous_DES = DESRegression(regressors_list = [Lasso(alpha = 0.15),SVR()], 
     n_estimators_bag = 10, 
     DSEL_perc = 0.95, 
     XTRAIN_full = True, 
     distance = distance.minkowski, 
     competence_region = 'output_profiles', 
     competence_level= mean_squared_error)

heterogeneous_DES.fit(X_train, y_train)
y_pred = heterogeneous_DES.predict(X_test)
print('MSE error:', mean_squared_error(y_test, y_pred))
