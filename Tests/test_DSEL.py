import unittest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from desReg.des.DESRegression import DESRegression


class TestDESRegression(unittest.TestCase):

    def setUp(self):
        # Create an instance of DesRegression with example parameters
        self.X, self.y = make_regression(n_samples=1000, n_features=10, random_state=0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)
        

    def test_des_regression(self):
        # Test DES regression by fitting the model, validating parameters, and predicting on the test set
        clf = DESRegression(k=3)
        clf.fit(self.X_train, self.y_train)
        clf._validate_parameters()
        y_pred = clf.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test))
        

    def test_competence_region_cluster(self):
        # Test competence region calculation using the cluster method
        clf = DESRegression(k=3)
        clf.fit(self.X_train, self.y_train)
        clf._validate_parameters()
        instance = self.X_test[0]
        instance = np.atleast_2d(instance)
        idxs, dists = clf._competence_region_cluster(instance)
        

    def test_competence_region_knn(self):
        # Test competence region calculation using the knn method
        clf = DESRegression(k=3)
        clf.fit(self.X_train, self.y_train)
        clf._validate_parameters()
        instance = self.X_test[0]
        instance = np.atleast_2d(instance)
        idxs, dists = clf._competence_region_knn(instance)
        self.assertEqual(len(idxs), clf.k)
        self.assertEqual(len(dists), clf.k)

    def test_competence_region_output_profiles(self):
        # Test competence region calculation using the output profiles method
        clf = DESRegression(k=3)
        clf.fit(self.X_train, self.y_train)
        clf._validate_parameters()
        instance = self.X_test[0]
        instance = np.atleast_2d(instance)
        idxs, dists = clf._competence_region_output_profiles(instance)
        self.assertTrue(len(idxs) > 0)
        self.assertTrue(len(dists) > 0)

if __name__ == '__main__':
    unittest.main()
