from desReg.des.DESRegression import DESRegression
from desReg.dataset import load_Student_Mark

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Lasso

from scipy.spatial import distance

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

heterogeneous_DES = DESRegression(
     regressors_list = [Lasso(alpha = 0.15),SVR()], 
     n_estimators_bag = 10, 
     DSEL_perc = 0.95, 
     XTRAIN_full = True, 
     distance = output_profile_distance, 
     competence_region = 'output_profiles', 
     competence_level= mean_squared_error
)

heterogeneous_DES.fit(X_train, y_train)
y_pred = heterogeneous_DES.predict(X_test)
print('MSE error:', mean_squared_error(y_test, y_pred))
