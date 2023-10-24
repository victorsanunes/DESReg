import unittest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from desReg.des.DESRegression import DESRegression

import desReg.utils.measures as em

class TestDESIntegration(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    # Combination: competence_region='knn', ensemble_type='DES', ensemble=defined
    def test_competence_region_knn_ensemble_type_DES(self):
        regressors = [LinearRegression(), Ridge(), DecisionTreeRegressor(), SVR()]
        des = DESRegression(competence_region='knn', k=5, ensemble_type='DES', regressors_list=regressors)
        des.fit(self.X_train, self.y_train)
        des._validate_parameters()
        y_pred = des.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        #Simple test to verify that the mse is calculated.
        self.assertLess(mse, 500)

    # Combination: competence_region='knn', ensemble_type='DRS', ensemble=defined
    def test_competence_region_knn_ensemble_type_DRS(self):
        regressors = [LinearRegression(), Ridge(), DecisionTreeRegressor(), SVR()]
        des = DESRegression(competence_region='knn', k=5, ensemble_type='DRS', regressors_list=regressors)
        des.fit(self.X_train, self.y_train)
        des._validate_parameters()
        y_pred = des.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, 500)

    # Combination: ensemble_type='SE', ensemble=defined
    def test_competence_region_knn_ensemble_type_SE(self):
        regressors = [LinearRegression(), Ridge(), DecisionTreeRegressor(), SVR()]
        des = DESRegression( k=5, ensemble_type='SE', regressors_list=regressors)
        des.fit(self.X_train, self.y_train)
        des._validate_parameters()
        y_pred = des.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, 500)


    # Combination: competence_region='cluster', ensemble_type='DES', ensemble=defined
    def test_competence_region_cluster_ensemble_type_DES(self):
        X_test = self.X_test.reshape(-1, 1)
        regressors = [LinearRegression(), Ridge(), DecisionTreeRegressor(), SVR()]
        des = DESRegression(competence_region='cluster', k=5, ensemble_type='DES', regressors_list=regressors)
        des.fit(self.X_train, self.y_train)
        des._validate_parameters()
        X_test = X_test.reshape(-1, 1)
        y_pred = des.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, 500)

    # Combination: competence_region='cluster', ensemble_type='DRS', ensemble=defined
    def test_competence_region_cluster_ensemble_type_DRS(self):
        X_test = self.X_test.reshape(-1, 1)
        regressors = [LinearRegression(), Ridge(), DecisionTreeRegressor(), SVR()]
        des = DESRegression(competence_region='cluster', k=5, ensemble_type='DRS', regressors_list=regressors)
        des.fit(self.X_train, self.y_train)
        des._validate_parameters()
        X_test = X_test.reshape(-1, 1)
        y_pred = des.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, 500)



    # Combination: competence_region='output_profiles', ensemble_type='DES', ensemble=defined
    def test_competence_region_output_profiles_ensemble_type_DES(self):
        X_test = self.X_test.reshape(-1, 1)
        regressors = [LinearRegression(), Ridge(), DecisionTreeRegressor(), SVR()]
        des = DESRegression(competence_region='output_profiles',k=5, ensemble_type='DES', regressors_list=regressors)
        des.fit(self.X_train, self.y_train)
        des._validate_parameters()
        X_test = X_test.reshape(-1, 1)
        y_pred = des.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, 500)

    # Combination: competence_region='output_profiles', ensemble_type='DRS', ensemble=defined
    def test_competence_region_output_profiles_ensemble_type_DRS(self):
        X_test = self.X_test.reshape(-1, 1)
        regressors = [LinearRegression(), Ridge(), DecisionTreeRegressor(), SVR()]
        des = DESRegression(competence_region='output_profiles',k=5, ensemble_type='DRS', regressors_list=regressors)
        des.fit(self.X_train, self.y_train)
        des._validate_parameters()
        
        y_pred = des.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, 500)
        


    # Combination: no ensemble specified, ensemble_type='DES'
    def test_no_competence_region_ensemble_type_DES(self):
        des = DESRegression(ensemble_type='DES')
        des.fit(self.X_train, self.y_train)
        des._validate_parameters()
        y_pred = des.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, 500)

    # Combination: competence_region='knn', ensemble_type='DES', competence_level=variance_error
    def test_competence_region_knn_ensemble_type_DES_selection_var(self):
        regressors = [LinearRegression(), Ridge(), DecisionTreeRegressor(), SVR()]
        des = DESRegression(competence_region='knn', k=3, ensemble_type='DES', regressors_list=regressors,competence_level=em.variance_error)
        des.fit(self.X_train, self.y_train)
        des._validate_parameters()
        y_pred = des.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, 500)

    #Combination: competence_region='knn', ensemble_type='DES', competence_level=variance_error, aggregation_method=weighted_average
    def test_competence_region_knn_ensemble_type_DES_selection_var_wa(self):
        regressors = [LinearRegression(), Ridge(), DecisionTreeRegressor(), SVR()]
        des = DESRegression(competence_region='knn', k=3, ensemble_type='DES', regressors_list=regressors,competence_level=em.variance_error,aggregation_method=em.weighted_average)
        des.fit(self.X_train, self.y_train)
        des._validate_parameters()
        y_pred = des.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.assertLess(mse, 600)



if __name__ == '__main__':
    unittest.main()




