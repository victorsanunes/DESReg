import unittest
import numpy as np
from desReg.dataset import load_Student_Mark
from sklearn.model_selection import train_test_split



from desReg.des.DESRegression import DESRegression

class TestDESRegression(unittest.TestCase):

    def setUp(self):
        self.data = load_Student_Mark()
        self.X = self.data.iloc[:,:-1].to_numpy()
        self.y = np.ravel(self.data.iloc[:, -1:])
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
      
        self.des = DESRegression()

    def test_fit_predict(self):
        
        self.des.fit(self.X_train, self.y_train)

        y_pred = self.des.predict(self.X_test)

        self.assertEqual(len(y_pred), len(self.y_test))

    def test_ensemble_type(self):
        self.des = DESRegression(ensemble_type='DRS')
        self.assertEqual(self.des.ensemble_type, 'DRS')

    def test_regressor_selection(self):
        self.des = DESRegression(regressor_selection=np.median)
        self.assertEqual(self.des.regressor_selection, np.median)

if __name__ == '__main__':
    unittest.main()
