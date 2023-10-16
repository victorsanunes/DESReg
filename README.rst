.. image:: https://img.shields.io/pypi/pyversions/scikit-learn.svg
.. :target: https://pypi.org/project/scikit-learn/

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://opensource.org/license/mit/

DESReg
======

Dynamic Ensemble Selection for Regression. This library implements the latest techniques to easily design and run Dynamic Ensemble Selection (DES) for regression tasks.  DESReg is based on scikit-learn_, and uses the same method signatures: **fit** and **predict**.

State of art dynamic techniques in dynamic selection [1]_ [2]_ can be accessed thanks to the high configurability of DESReg. allows to test existing methodologies or to design new ones by configuring them from the establishment of the corresponding hyperparameters. In addition, DESReg the library makes it easy for the user to design new techniques and pass them as new parameter values. Other features of the package include the ability to test classical ensemble methodologies or its support for parallelization of key stages in the execution of ensembles.

Modules
------------
The package includes two folders with the following contents:

1. DESReg.desReg:  
    - des.DESRegression: Implementation of all ensemble functionalities. 
    - utils.measures: Implementation of various error and aggregation measures for the base regressors.
2. DESReg.examples: Examples with code for creating and using regression ensembles. 


Parameters and configuration
----------------
The constructor method ``DESRegression`` accepts the following parameter to design a new DES methodology.

- ``regressors_list:`` Accepts a list with the regressors that will be part of the ensemble. The pool of regressors is obtained by bagging each of the regressors in the list. The base regressors must support the ``predict`` method. If this list is not defined, the default regressor for bagging is the ``DecisionTreeRegressor``.
          
- ``n_estimators_bag:`` Number of instances of each base regressor for bagging. The total number of base regressors in the ensemble is ``n_estimator_bag``x``len(regressors_list)``.
- ``DSEL_perc:`` The proportion of samples making up the DSEL. The default value is set to 0.5.
- ``XTRAIN_full``: Determines whether the training set will be fully used. Otherwise, it will be the complementary set to the DSEL. True is its default value.
- ``n_jobs``: Number of jobs to run in parallel the fitting or prediction, with bagging (``BaggingRegressor``), of the regressors pool. Also, it is used, with the same meaning in the neighbors search of the k-NN method (``KNeighborsRegressor``).
- ``k:`` Number of neighbors or clusters for estimating the region of competence. Its default value is 5. 
- ``distance:`` Accepts a function that calculates the distance between two 1-D arrays. This function will be used for the calculation of all necessary distances. By default euclidean (``scipy.spatial.distance.euclidean``) distance is used.
- ``competence_region:`` Receives a string that determines the type of the region of competence to calculate. Their possible values are: 'KNN', 'cluster' or 'profiles'.
- ``competence_level:`` Accepts a function or list of functions that obtain an error (or list of errors) of the regressor over the competence region. 
- ``regressor_selection:`` Receives a method that calculates the selection threshold to choose the best regressors (with the best competence levels) from the pool. The arithmetic mean (``numpy.mean``) of the competence level regressors is used by default. In this manner, any regressor with a competence level below the threshold is selected. As for the above hyperparameters, the user can utilize other already implemented measures or implements its own measures.
- ``aggregation_method:`` Given method for aggregating the predictions of the ensemble base regressors. 
- ``ensemble_type:`` Determine the type of ensemble. Posible values are: DES (Dynamic Ensemble Selection), DSR (Dynamic Regressor Selecion), SE (Static Ensemble)
Installation:
-------------

The library can be installed using ``pip``:

.. code-block:: bash

    pip install DESreg


Dependencies
-------------------
DESReg has been tested with Python 3.8.5. The dependency requirements are:

* scikit-learn(>=1.2.1)
* numpy(>=1.21.5)
* scipy(>=1.5.2)


These dependencies are automatically installed when the pip command is used.



Example
-----------
Basis use of DESReg:

.. code-block:: python

	from desReg.des.DESRegression import DESRegression
	from desReg.dataset import load_Student_Mark

	import numpy as np

	from sklearn.model_selection import train_test_split
	from sklearn.metrics import mean_squared_error


	data = load_Student_Mark()
	X = data.iloc[:,:-1].to_numpy()
	y = np.ravel(data.iloc[:, -1:]) 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

	# DES declaration
	homogeneous_DES = DESRegression()
	# DES fitting
	homogeneous_DES.fit(X_train, y_train)
	# DES prediction
	y_pred = homogeneous_DES.predict(X_test)
	print('MSE error:', mean_squared_error(y_test, y_pred))


Citation
---------

If you use DESReg in a scientific paper, please consider citing the following paper:

References:
---------
.. [1] : R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,” Information Fusion, vol. 41, pp. 195 – 216, 2018.
.. [2] : Thiago J.M. Moura and George D.C. Cavalcanti and Luiz S. Oliveira, "MINE: A framework for dynamic regressor selection",Information Sciences, vol. 543, pages 157-179, 2021.
.. _scikit-learn: http://scikit-learn.org/stable/.. _scikit-learn: http://scikit-learn.org/stable/
