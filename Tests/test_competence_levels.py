import unittest
from desReg.des.DESRegression import DESRegression
import numpy as np

import desReg.utils.measures as em

class TestDESRegression(unittest.TestCase):

    def setUp(self):
        # Set up the test data and initialize the DESRegression instance
        self.X = np.array([[1,2], [3,4], [5,6], [7,8], [9,10], [11,12]])
        self.y = np.array([0, 1, 2, 3, 4, 5])
        self.clf = DESRegression(k=3)
        self.clf.fit(self.X, self.y)
        self.clf._validate_parameters()
        

    def test_calculate_competence_level_returns_array_of_correct_shape(self):
        #Test if the matrix returned for the competence levels matches with the shape that the regressor matrix.
        idxs = np.array([0, 1, 2])
        dists = np.array([0.1, 0.2, 0.3])
        instance = self.X[0]
        instance = np.atleast_2d(instance)
        print(instance)
        competence_levels = self.clf._calculate_competence_level(idxs, dists, instance)
        self.assertEqual(competence_levels.shape[0], len(self.clf.regressors_list_))

    def test_calculate_competence_level_returns_value_between_zero_and_one(self):
        # Test if the value for the competence level for each regressor is between 0 and 1.
        idxs = np.array([0, 1, 2])
        dists = np.array([0.1, 0.2, 0.3])
        instance = self.X[0]
        instance = np.atleast_2d(instance)
        competence_levels = self.clf._calculate_competence_level(idxs, dists, instance)
        self.assertTrue(all(0 <= c <= 1 for c in competence_levels))

    def test_calculate_competence_level_returns_coherent_results_for_different_error_measures(self):
        # Test if the results are coherent for different error measures.
        idxs = np.array([0, 1, 2])
        dists = np.array([0.1, 0.2, 0.3])
        instance = self.X[0]
        instance = np.atleast_2d(instance)
        competence_levels1 = self.clf._calculate_competence_level(idxs, dists, instance)
        self.clf.competence_level_ = [em.variance_error]
        competence_levels2 = self.clf._calculate_competence_level(idxs, dists, instance)
        self.assertFalse(np.array_equal(competence_levels1, competence_levels2))

    def test_calculate_competence_level_returns_error_if_invalid_instance(self):
        # Test if the function raises the correct error when an invalid competence level is indicated.
        idxs = np.array([0, 1, 2])
        dists = np.array([0.1, 0.2, 0.3])
        instance = self.X[0]
        instance = np.atleast_2d(instance)
        self.clf.competence_level_=["invalid"]
        with self.assertRaises(TypeError):
            self.clf._calculate_competence_level(idxs, dists, instance)

if __name__ == '__main__':
    unittest.main()
