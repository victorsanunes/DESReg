from inspect import signature
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.utils.validation import (check_X_y, check_array,
                                     check_random_state)
from sklearn.utils.validation import check_array
from sklearn.preprocessing import MinMaxScaler


from scipy.spatial import distance

import desReg.utils.measures as em


MIN_NUMBER_INSTANCES = 5 # minimum number of instances in the dataset
MIN_N_REGRESSORS_BAG = 2  # minimum number of estimators
COMPETENCE_REGION_TYPES = ['knn', 'cluster', 'output_profiles']
ENSEMBLE_TYPES = ['DES', 'DRS', 'SE']


class DESRegression(BaseEstimator):    
    """Class for a Dynamic Ensemble Selection (DES) for regression
       This class can also be use for building a Dynamic Regressor Selection (DRS) and a Static Ensemble (SE)
    """

    def __init__(self, regressors_list=None, n_estimators_bag=10, random_state=None, DSEL_perc=0.5, XTRAIN_full=True, n_jobs=-1,
                    k=5, distance = distance.euclidean, competence_region='knn', competence_level= em.all_errors, regressor_selection=np.mean, 
                    aggregation_method=np.mean, ensemble_type='DES'):   
              
        """
           Constructor method

        Parameters
        ----------
        regressors_list : list (Default = None)
            List with the regressors that will be part of the ensemble. 
            The pool of regressors is obtained by bagging each regressor in the list. 
            The base regressors must support the predict method.
            If none, the default regressor for baggin is DecisionTreeRegressor. 

        n_estimators_bag : int, (Default = 10)
            Number of base regressors in each bagging. 
            The total number of base regressors in the pool is n_estimator_bag*len(regressors_list).

        random_state : int, (Default = None)
            Seed to generate random numbers. 
            If none, a random seed is generated.

        DSEL_perc : float, (Defalult = 0.5)
            The proportion of samples making up the DSEL, a real between 0.0 and 1.0.

        XTRAIN_full : bool (Default = True)
            Determines whether the training set is full taken. Otherwise, it will be the complementary set to the DSEL.

        n_jobs : int, (Default = -1)
            Number of jobs to run in parallel for methods such as fit or predict (with bagging) or to run neighbour search. 
            If none, the value is set to -1, so that all processors can be used. 

        k : int, (Default = 5)
            Number of neighbours or clusters for estimating the region of competence. 

        distance: callable, (Default = distance.euclidean)
            Function used for the calculation of distances between two 1-D arrays.

        competence_region : str, (Default = 'knn')
            Determines the calculation of the region of competence for a given instance. 
            Possible values are: 'knn', 'cluster' and 'output_profiles'.       
            
        competence_level : callable or callable list, (Default = nm.all.errors)
            Accepts a function or list of functions that returns an error or list of errors. 
            The mean of these errors determines the competence level of the regressors of the ensemble. 
            
        regressor_selection : callable, (Defaul = np.mean)
            Method used to determine the selection threshold to choose the best regressors.

        aggregation_method : callable, (Defalul = np.mean)
            Method for aggregating the predictions of the ensemble base regressors. 
            It can accept weigthed aggregations.

        ensemble_type: str,  (Default = 'DES')
            Determine the type of ensemble.
            Posible values are: 'DES' (Dynamic Ensemble Selection), 'DSR' (Dynamic Regressor Selecion), 'SE' (Static Ensemble)
   
        """                                            
        self.regressors_list = regressors_list
        self.n_estimators_bag = n_estimators_bag
        self.random_state = random_state
        self.DSEL_perc = DSEL_perc
        self.XTRAIN_full = XTRAIN_full
        self.n_jobs = n_jobs  
        self.k = k
        self.distance = distance
        self.competence_region = competence_region
        self.competence_level = competence_level
        self.regressor_selection = regressor_selection
        self.aggregation_method = aggregation_method
        self.ensemble_type = ensemble_type
        
  
    def _set_dsel(self, X, y):
                    
        """Set up the structure of the dynamic selection dataset (DSEL) from the input data
        and validate some of its parameters
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data

        y : array of shape (n_samples)
            Output associated with each instance of X.
           
        """
               
        self.DSEL_data_ = X
        self.DSEL_target_ = y
        self.n_features_ = X.shape[1]  
        self.n_samples_ = self.DSEL_target_.size

    def _predict_base(self, X):
                
        """Get the predictions of each base estimator in the pool for all
            samples in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data

        Returns
        -------
        array of shape (n_samples, n_regressors): 
             The predictions of each base estimator for all samples in X
        """        
        
        predictions = []
        for reg in self.regressors_list_:
            predicted_outputs = reg.predict(X)
            predictions.append(predicted_outputs) 
        predictions = np.array(predictions)  
        predictions = np.transpose(predictions)  
        return predictions 
      

  
    def _validate_parameters(self): 
        """  Verify if several input parameters are correct (k, distance, competence_region, competence_level, regressor_selection, aggregation_method 
              and ensemble_type) 

        Raises
        ------
        TypeError
            If the parameter k is not an integer
        ValueError
            If the parameter k is less than 1 or larger than the size of the DSEL
        TypeError
            If the parameter distance is not callable
        ModuleNotFoundError
            If the parameter competence_region is not a string
        ValueError
            If the parameter competence_region not in COMPETENCE_REGION_TYPES
        ModuleNotFoundError
            If competence_level is not callable
        ModuleNotFoundError
            If regressor_selection is not callable
        ModuleNotFoundError
            If aggregation_method is not callable
        TypeError
            If the parameter ensemble_type is not a string
        ValueError
            If the parameter ensemble_type not in ENSEMBLE_TYPES
        """                      
        
        if not isinstance(self.k, int):
            raise TypeError('Parameter k should be an integer')
        elif self.k > self.n_samples_ or self.k <= 1:
            raise TypeError('k must be greater than 1 and less than the number of samples in the DSEL. k is to be used as len(DSEL)-1')
        else:  
            self.k_ = self.k
        
        if not callable(self.distance):
            raise ModuleNotFoundError(self.distance, 'method is not defined')
        else:
            self.distance_ = self.distance 

        if not isinstance(self.competence_region, str):
            raise TypeError('Parameter competence_region should be a string')
        elif self.competence_region not in COMPETENCE_REGION_TYPES:  
            raise ValueError(f'Error in competence_region parameter. Its possible values are: {COMPETENCE_REGION_TYPES}')
        else:
            self.competence_region_ = self.competence_region
                
        type(self.competence_level)
        if (type(self.competence_level) is not list):
            self.competence_level = [self.competence_level]
        for func in self.competence_level:
            if not callable(func):
                raise ModuleNotFoundError(func, 'method is not defined')
        self.competence_level_ = self.competence_level


        if not callable(self.regressor_selection):
            raise ModuleNotFoundError(self.regressor_selection, 'method is not defined')
        else:
            self.regressor_selection_ = self.regressor_selection
         
        if not callable(self.aggregation_method):
            raise ModuleNotFoundError(self.aggregation_method, 'method is not defined')
        else:
            self.aggregation_method_ = self.aggregation_method
        
        if not isinstance(self.ensemble_type, str):
            raise TypeError('Parameter ensemble_type should be a string')
        elif self.ensemble_type not in ENSEMBLE_TYPES:  
            raise ValueError(f'Error in ensemble_type parameter. Its possible values are: {ENSEMBLE_TYPES}')
        else:
            self.ensemble_type_ = self.ensemble_type
        


    def fit(self, X, y):   
        
        """Fit the model with the given parameters. First, validate some parameters such as the DSEL_perc, n_estimators_bag and then
        create the pool of regressors and the DSEL

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data

        y: array of shape (n_samples)
            Outputs of each sample in X 
        
        Returns
        -------
        self: object
            Object fitted 

        Raises
        ------
        ValueError
            If the number of instances in the dataset is not at least MIN_NUMBER_INSTANCES
        ValueError
            If the parameter DSEL_perc is not in range [0.1, 1]
        ValueError
            If the parameter n_estimators_bag is less than MIN_N_REGRESSORS_BAG
        
        """

        if len(X) < MIN_NUMBER_INSTANCES:  
            raise ValueError("The number of instances in the dataset must be at least ", MIN_NUMBER_INSTANCES)       
        self.random_state_ = check_random_state(self.random_state)
        X, y = check_X_y(X, y)

        if self.DSEL_perc <0.1 or self.DSEL_perc > 1.0:
            raise ValueError('DSEL_perc must be a value between 0.1 and 1.0')
        else:
            self.DSEL_perc_ = self.DSEL_perc
        
        if self.DSEL_perc_ == 1.0:
            X_dsel = X
            y_dsel = y
            X_train = X
            y_train = y
        else:
            X_train, X_dsel, y_train, y_dsel = train_test_split(
                    X, y, test_size=self.DSEL_perc_, random_state=self.random_state_)
        
        if self.XTRAIN_full == True:  
            X_train = X
            y_train = y    
        

        if  self.n_estimators_bag < MIN_N_REGRESSORS_BAG:
            raise ValueError("Number of estimators_bag must be at least ", MIN_N_REGRESSORS_BAG)
        else:
            self.n_estimators_bag_ = self.n_estimators_bag

        
        self.n_jobs_ = int(self.n_jobs)
        
        if self.regressors_list is None:
            self.regressors_list_ = BaggingRegressor(
                random_state = self.random_state_, n_estimators = self.n_estimators_bag_, n_jobs = self.n_jobs_).fit(X_train, y_train)
        else:
            new_pool = []
            for regressor in self.regressors_list:
                new_pool.append(BaggingRegressor(
                        estimator=regressor, random_state=self.random_state_, n_estimators = self.n_estimators_bag_, n_jobs=self.n_jobs_).fit(X_train, y_train))

            self.regressors_list_ = []
            for bag in new_pool:
                for regressor in bag:
                    self.regressors_list_.append(regressor)

        self.n_regressors_ = len(self.regressors_list_)
       
        self._set_dsel(X_dsel, y_dsel)

        return self
    
    
###########################################################################
###   Define the region of competence (DSEL: Dynamic Selection Dataset) ###
###########################################################################

    def _competence_region_cluster(self, instance):
        
        """Calculate the region of competence using clustering

        Parameters
        ----------
        instance : array of real
            Sample of the test dataset

        Returns
        -------
        array of int:
            Indices of the DSEL samples forming the region of competence

        array of real:
            Distances between instance and the samples in the region of competence    
        """
        
        self.clustering_ = KMeans(n_clusters=self.k_, random_state=self.random_state_)  

        self.clustering_.fit(self.DSEL_data_)
        
        clust = self.clustering_.predict(instance)
        idxs = []
        for i in range(0, len(self.DSEL_data_)):
            if self.clustering_.predict(self.DSEL_data_[i].reshape(1, -1)) == clust:
                idxs.append(i)
        
        dists = np.zeros(len(idxs))
        for i in range(0, len(idxs)):
            dists[i] = self.distance_(self.DSEL_data_[idxs[i]], instance)

        return idxs, dists

    def _competence_region_knn(self, instance):
        
        """Calculate the region of competence using k-NN method

        Parameters
        ----------
        instance : array of real
            Sample of the test dataset

        Returns
        -------
        array of int:
            Indices of the DSEL samples forming the region of competence

        array of real:
            Distances between instance and the samples in the region of competence    
        """

 
        self.knn_= KNeighborsRegressor(n_neighbors=self.k_, algorithm='auto', metric = self.distance_, n_jobs=self.n_jobs_).fit(self.DSEL_data_, self.DSEL_target_)
        dists, idxs = self.knn_.kneighbors(instance, n_neighbors=self.k_, return_distance=True)  # por defecto metric='minkowski'??????
        
        idxs = idxs.reshape(-1)
        dists = dists.reshape(-1)
                     
        return idxs, dists


    def _competence_region_output_profiles(self, instance):
        
        """Calculates the region of competence on the basis 
        of the output profile method

        Parameters
        ----------
        instance : array of real
            Sample of the test dataset

        Returns
        -------
        array of int:
            Indices of the DSEL samples forming the region of competence

        array of real:
            Distances between instance and the samples in the region of competence    
        """

        
        output_profile_instance = []

        self.pred_DSEL_ = self._predict_base(self.DSEL_data_)
               
        for regressor in self.regressors_list_:
            output_profile_instance.append(regressor.predict(instance))

        idxs = []
        idx_count = 0
    
        similarity_vector = np.zeros(len(self.pred_DSEL_))

        for dsel_idx in self.pred_DSEL_:  
            similarity = self.distance_(dsel_idx, output_profile_instance)

            similarity_vector[idx_count] = similarity
            idx_count += 1

        umbral = np.mean(similarity_vector)
        idx_count = 0
        for similarity in similarity_vector:
            if similarity < umbral:
                idxs.append(idx_count)
            idx_count += 1

        dists = np.zeros(len(idxs))           
        for i in range(0, len(idxs)):
            dists[i] = self.distance_(self.DSEL_data_[idxs[i]], instance)
        
        return idxs, dists


######################################################################
###    Calculating the level of competence of the base regressors  ###
######################################################################
    
    def _calculate_competence_level(self, idxs, dists, instance):
        
        """Calculate the competence level of each regressor 
        in the pool for the given test sample

        Parameters
        ----------
        idxs : array of int
            Indices of the DSEL samples forming the region of competence

        dists : array of real
            Distances between instance and the samples in the region of competence
        
        instance : array of real
            Sample of the test dataset

        Returns
        -------
        array of real
            Competence level of each regressor in the pool for the given test sample

        """
               
        selected_measures = self.competence_level_
        n_measures = len(selected_measures)
        if em.all_errors in self.competence_level_:
            n_measures = n_measures + em.N_ERROR_MEASURES -1

        regressors_errors = np.empty((0, n_measures))
        
        competence_region = self.DSEL_data_[idxs]
        dists[dists == 0] = 1e-10
        inverse_distances = 1.0 / dists
        sum_inverse_distances = np.sum(inverse_distances)
        normalized_distances = inverse_distances/sum_inverse_distances
        
        y_true = self.DSEL_target_[idxs]
        for reg in self.regressors_list_:
            errors = []
            y_pred = reg.predict(competence_region)
            y_pred_test = reg.predict(instance) 
            for func in selected_measures:
                params = signature(func).parameters
               
                l = []
                for param in params: 
                    if param in locals():
                        l.append((param,locals()[param]))
                wargsk = dict(l) 
                error = func(**wargsk)   
                
                errors = np.append(errors, error)
            regressors_errors = np.append(regressors_errors, [errors], axis = 0)
        
        scaler = MinMaxScaler()
        regressors_errors = scaler.fit_transform(regressors_errors)

        competence_levels = np.mean(regressors_errors, axis = 1) 
        return competence_levels
   
###########################################################
###   Dynamic selection of regressors                   ###
###########################################################

    def _select_regressors_DRS(self, competence_levels):

        """Return the best regressor in the pool

        Parameters
        ----------
        competence_levels : array of shape (1, n_regressors_pool)
            level of competence (error) of each regressor in the pool
            
        Returns
        -------
        regresor
            The best regressor of the pool

        """

        indx_best_regressor = np.where(competence_levels == np.min(competence_levels)) # find the regressor with the minimum error
        indx_best_regressor = indx_best_regressor[0][0]   # if there are several with the same level of competence, the first on the list is selected
        best_regressor = self.regressors_list_[indx_best_regressor]
        return best_regressor
    
    
    def _select_regressors_DES(self, competence_levels):  
        
        """Return the ensemble of regressors with their competence levels

        Parameters
        ----------
        competence_levels : array of shape (1, n_regressors_pool)
            level of competence (error) of each regressor in the pool
            
        Returns
        -------
        array
            The best regressors of the pool

        array
            Level of competence of the selected regressors 
        """

        umbral = self.regressor_selection_(competence_levels)
                
        selected_regressors = []
        selected_competence_levels = []
        
        for reg in range(0,self.n_regressors_): 
            if competence_levels[reg] <= umbral: 
                selected_regressors.append(self.regressors_list_[reg])
                selected_competence_levels.append(competence_levels[reg])

        return selected_regressors, np.array(selected_competence_levels)

    def select_regressors_SE(self): 
        
        """Return an ensemble with all the regressors 

        Returns
        -------
        array
            regressors of the pool
        """
        return self.regressors_list_ 
            
        
  
###########################################################
###          Aggregation   or combination               ###
###########################################################
   
    def _aggregation(self, ensemble, instance, competence_levels):

        """Aggregate the predictions of each regressor in ensemble 
        for a given sample.
        
        Parameters
        ----------
        ensemble : list
            selected regressors of the pool
        instance : array 
            instance to predict
        competence_levels : array shape (1, n_regressors_ensemble)
            level of competence (error) of each regressor in the ensemble

        Returns
        -------
        real 
            Ensemble prediction for the given sample
        """
            
        instance = check_array(instance, ensure_2d=False)
        if instance.ndim == 1:
            instance = np.atleast_2d(instance)

        predictions = np.array([])
        for reg in ensemble:
            predictions = np.append(predictions, reg.predict(instance))

        competence_levels[competence_levels == 0] = 1e-10
        weights = 1.0 / competence_levels 
        weights = np.array(weights/np.sum(weights)) 
                
        sig = signature(self.aggregation_method_)
        params = sig.parameters
        
        if 'weights' in params: 
            prediction = self.aggregation_method_(predictions, weights=weights )  
        else:
            prediction = self.aggregation_method_(predictions)  

        return prediction
         
###########################################################
###          Prediction                                 ###
###########################################################
    
    def predict(self, X, params=None):

        """Calculate sample predictions using Dinamic Ensemble Selection

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Samples
        params: dictionary 
            Name and value of the parameters to be changed. The regressor pool and DSEL remain unchanged. 
        If None the parameter values are those given when the ensemble object is created 
        
        Returns
        -------
        array of shape(n_samples)
            The prediction of the ensemble for each sample in X
        """
        if not params is None:  
            self.set_params(**params)
        
        params = self.get_params(deep=False) 
        
        self._validate_parameters()
        
        name_func = '_predict_' + self.ensemble_type_  
        func = getattr(self, name_func)
            
        y_pred = func(X)  
                     
        return y_pred  

    def _predict_DES(self, X):

        """Calculate sample predictions using Dinamic Ensemble Selection

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Samples
        
        Returns
        -------
         array of shape(n_samples)
            The prediction of the ensemble for each sample in X
        """
        y_pred = []
        for instance in X:
            instance = np.atleast_2d(instance)

            name_func = '_competence_region_' + self.competence_region_  
            func = getattr(self, name_func)
            idxs, dists  = func(instance)             
            competence_levels = self._calculate_competence_level(idxs, dists, instance)
            selected_regressors, selected_competence_levels = self._select_regressors_DES(competence_levels) 
            ensemble = selected_regressors
            predicted_output = self._aggregation(ensemble, instance, selected_competence_levels) 
            y_pred.append(predicted_output)
        
        return y_pred  

    def _predict_DRS(self, X):  
        
        """Calculate sample predictions using the best ensemble regressor

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Samples
        
        Returns
        -------
         array of shape(n_samples)
            The prediction of the best regresor for each sample in X
        """

        y_pred = []
        for instance in X:
            instance = np.atleast_2d(instance)
            name_func = '_competence_region_' + self.competence_region_  
            func = getattr(self, name_func)
            idxs, dists  = func(instance)  
            competence_levels = self._calculate_competence_level(idxs, dists, instance) 
            selected_regressor = self._select_regressors_DRS(competence_levels) 
            predicted_output = selected_regressor.predict(instance).reshape(-1)
            y_pred.append(predicted_output)
        return y_pred  
    
    def _predict_SE(self, X):  
        
        """Calculate sample predictions using the static ensemble methodology. All regressors in the pool are used for prediction

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Samples
        
        Returns
        -------
         array of shape(n_samples)
            The prediction of the static ensemble for each sample in X
        """
         
        y_pred = []
        ensemble = self.select_regressors_SE()
        predictions = []
        for reg in ensemble:
            predictions.append(reg.predict(X)) 
        
        y_pred = np.mean(predictions, axis = 0)
        
        return y_pred
    
