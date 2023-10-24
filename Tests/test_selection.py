import unittest
import numpy as np
from desReg.des.DESRegression import DESRegression
from sklearn.svm import SVR

class TestDesRegression(unittest.TestCase):

    def setUp(self):
        # Create an instance of DesRegression with example parameters
        self.X = np.array([[1,2], [3,4], [5,6], [7,8], [9,10], [11,12]])
        self.y = np.array([0, 1, 2, 3, 4, 5])
        self.pool_regressors = [SVR()]
        self.des = DESRegression(self.pool_regressors,k=3)
        self.des.fit(self.X,self.y)
        self.des._validate_parameters()
        
    def test_select_regressors_DRS(self):
        # Test if the best regressor is returned by select_regressors_DRS
        competence_levels = np.array([0.1, 0.2, 0.05, 0.3, 0.15,0.1, 0.2, 0.05, 0.3, 0.15])
        best_regressor = self.des._select_regressors_DRS(competence_levels)
        self.assertEqual(best_regressor, self.des.regressors_list_[2])
        
    def test_select_regressors_DES(self):
        # Test if the competent regressors are correctly selected by select_regressors_DES
        competence_levels = np.array([0.1, 0.2, 0.05, 0.3, 0.15,0.1, 0.01, 0.05, 0.02, 0.15])
        selected_regressors, selected_competence_levels = self.des._select_regressors_DES(competence_levels)
        self.assertEqual(selected_competence_levels.tolist(), [0.1, 0.05, 0.1, 0.01, 0.05, 0.02])
        
    def test_select_regressors_SE(self):
        # Test if all the regressors from the pool are returned by select_regressors_SE
        all_regressors = self.des.select_regressors_SE()
        self.assertEqual(all_regressors, self.des.regressors_list_)
        
if __name__ == '__main__':
    unittest.main()
