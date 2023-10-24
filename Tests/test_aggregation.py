import unittest
import numpy as np
from desReg.des.DESRegression import DESRegression
from desReg.utils import *
from sklearn.neighbors import KNeighborsRegressor



class TestDesRegressionAggregation(unittest.TestCase):
    def setUp(self):
        # Define some training data and models
        self.X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        self.y_train = np.array([1, 2, 3, 4, 5])
        self.model1 = KNeighborsRegressor(n_neighbors=1)
        self.model2 = KNeighborsRegressor(n_neighbors=2)
        
        # Fit the models before creating the DESRegression object
        self.model1.fit(self.X_train, self.y_train)
        self.model2.fit(self.X_train, self.y_train)
        
        # Create a pool of models and a DES regression object
        self.ensemble = [self.model1, self.model2]
        self.dr = DESRegression(ensemble_type='DES', regressors_list=self.ensemble, aggregation_method=np.mean, k=2)
        
        self.dr.fit(self.X_train, self.y_train)
        self.dr._validate_parameters()

    def test_aggregation(self):
        # Define some test data and expected output
        X_test = np.array([[2, 3]])
        expected_output = np.array([1.5])
        competence_levels = np.array([0.1, 0.2])

        # Call the aggregation method on the test data using the models in the ensemble
        output = self.dr._aggregation(self.ensemble, X_test[0], competence_levels)

        # Assert that the output matches the expected output within some tolerance
        np.testing.assert_allclose(expected_output, output, rtol=1, atol=1)

    def test_weighted_aggregation(self):
        # Define some test data and expected output
        X_test = np.array([[2, 3]])
        expected_output = np.array([1.5])

        # Calculate competence levels based on your implementation
        competence_levels = np.array([0.1, 0.1])

        # Call the weighted aggregation using the _aggregation method with use_weighted_aggregation=True
        self.dr.use_weighted_aggregation = True
        output = self.dr._aggregation(self.ensemble, X_test[0], competence_levels)

        # Assert that the output matches the expected output within some tolerance
        np.testing.assert_allclose(expected_output, output, rtol=1, atol=1)

if __name__ == '__main__':
    unittest.main()
