import numpy as np



N_ERROR_MEASURES = 8   # method of error implemented, number of errors calculated from all_errors 

### Measures to estimate errors of the regressors for measuring the level of competence
def variance_error(y_pred):
    """Calculates the variance of the DSEL predictions 

    Parameters
    ----------
    y_pred : array
        actual output of samples from the competence region

    Returns
    -------
    
    real
        The variance of the input values 
    """
        
    return np.var(y_pred)

def sum_absolute_error(y_true, y_pred, normalized_distances):
    """_summary_

    Parameters
    ----------
    y_true : array
        actual output of samples from the competence region
    y_pred : array
        output predictions of a regressor for the samples of the competence region
    normalized_distances : array
        Inverse of the distance between the test sample and the samples in the competence region

    Returns
    -------
    real 
        The dot product between the absolute error and the normalized_distances
    """
    return np.dot(np.abs(y_true - y_pred), normalized_distances) # sum_absolute_error

def sum_squared_error(y_true, y_pred, dists):

    """_summary_

    Parameters
    ----------
    y_true : array
        actual output of samples from the competence region
    y_pred : array
        output predictions of a regressor for the samples of the competence region
    normalized_distances : array
       Inverse of the distance between the test sample and the samples in the competence region

    Returns
    -------
    real
        The dot product between the absolute error and the normalizeed_distances
    """

    return np.dot(np.square(y_true - y_pred), dists)  # sum square error

def minimum_squared_error(y_true, y_pred, normalized_distances):

    """_summary_

    Parameters
    ----------
    y_true : array
        actual output of samples from the competence region
    y_pred : array
        output predictions of a regressor for the samples of the competence region
    normalized_distances : array
       Inverse of the distance between the test sample and the samples in the competence region

    Returns
    -------
    real
        The minimum square error of the product between the error and the normalizeed_distances
    """
    return np.min(np.multiply(np.square(y_true - y_pred), normalized_distances)) # minimum_square_error

def maximum_squared_error(y_true, y_pred, normalized_distances):    

    """_summary_

    Parameters
    ----------
    y_true : array
        actual output of samples from the competence region
    y_pred : array
        output predictions of a regressor for the samples of the competence region
    normalized_distances : array
       Inverse of the distance between the test sample and the samples in the competence region

    Returns
    -------
    real
        The maximum square error of the product between the error and the normalizeed_distances
    """
    return np.max(np.multiply(np.square(y_true - y_pred), normalized_distances)) # maximum_square_error
                    
def neighbors_similarity(y_true, normalized_distances, y_pred_test):   
    """_summary_

    Parameters
    ----------
    y_true : array
        actual output of samples from the competence region
    normalized_distances : array
       Inverse of the distance between the test sample and the samples in the competence region    
    y_pred_test : array or real
        output prediction/s of a regressor for the test sample
    
    Returns
    -------
    real
        
    """ 
    return np.dot(np.square(y_true - y_pred_test), normalized_distances) # neighbors similarity ????Se vuelve a invertir la distancia????
                   
def root_sum_squared_error(y_true, y_pred, normalized_distances): 
    """_summary_

    Parameters
    ----------
    y_true : array
        actual output of samples from the competence region
    y_pred : array
        output predictions of a regressor for the samples of the competence region
    normalized_distances : array
       Inverse of the distance between the test sample and the samples in the competence region

    Returns
    -------
    real 
        The dot product between the root mean square error and the normalizeed_distances
    """   
    return np.sqrt(np.dot(np.square(y_true - y_pred), normalized_distances)) # root_sum_square_error
                    
    # error in the nearest sample
def nearest_squared_error(y_true, y_pred, dists):
    """_summary_

    Parameters
    ----------
    y_true : array
        actual output of samples from the competence region
    y_pred : array
        output predictions of a regressor for the samples of the competence region
    dists : array
       Distances between the test sample and the samples in the competence region

    Returns
    -------
    real
        The error in the sample of competence region closest to the test sample
    """ 
    p_min = np.where(dists == np.min(dists))
    p_min = p_min[0][0]  
    return np.square(y_true[p_min] - y_pred[p_min])  # closest_square_error
    
def all_errors(y_true, y_pred, normalized_distances, dists, y_pred_test):
    """_summary_

    Parameters
    ----------
    y_true : array
        actual output of samples from the competence region
    y_pred : array
        output predictions of a regressor for the samples of the competence region
    normalized_distances : array
       Inverse of the distance between the test sample and the samples in the competence region
    dists : array
       Distances between the test sample and the samples in the competence region
    y_pred_test : array or real
       output prediction/s of a regressor for the test sample

    Returns
    -------
    array
        An array with eight error measures of a regressor in the competition region
    """ 
    errors = []
    errors = np.append(errors, variance_error(y_pred))
    errors = np.append(errors, sum_absolute_error(y_true, y_pred, normalized_distances))
    errors = np.append(errors, sum_squared_error(y_true, y_pred, dists))
    errors = np.append(errors, minimum_squared_error(y_true, y_pred, normalized_distances))
    errors = np.append(errors, maximum_squared_error(y_true, y_pred, normalized_distances))
    errors = np.append(errors, neighbors_similarity(y_true, normalized_distances, y_pred_test))             
    errors = np.append(errors, root_sum_squared_error(y_true, y_pred, normalized_distances))
    errors = np.append(errors, nearest_squared_error(y_true, y_pred, dists))
    return errors
      
# Examples of functions that can be used as a method of aggregating predictions for the ensemble
# Some of these measures can also be used as a thresholding method in the selection of regressors for the ensemble

def mean_sd(a):
    """Value obtained as the arithmetic mean plus the standard deviation of the input values

    Parameters
    ----------
    a : array
        An array with the values for which the measure is to be calculated

    Returns
    -------
    real:
        The arithmetic mean of the values of a plus their standard deviation
    """   
    mean_error = np.mean(a)
    sd = a.std(ddof=0)
    return (mean_error + sd)



def max_min_2(a):

    """
    Parameters
    ----------
    a : array
        An array with the values for which the measure is to be calculated

    Returns
    -------
    real: half of the sum of the minimum and maximum of the vector
         
    """
    max_value = np.max(a)
    min_value = np.min(a)
    return ((max_value + min_value) /2)

def mean_min_2(a):
    """

    Parameters
    ----------
    a : array
        An array with the values for which the measure is to be calculated

    Returns
    -------
    real: half of the sum of the minimum and the mean of the vector
        
    """   
    mean_value = np.mean(a)
    min_value = np.min(a)
    return ((mean_value + min_value) /2)

def geometric_mean(a):

    """Calculates the geometric mean of the input values

    Parameters
    ----------
    a : array
        An array with the values from which its geometric mean is to be calculated

    Returns
    -------
    real:
        The geometric mean of the values of a 
    """
    p = 1
    for i in a:
        p = p*i
    return p**(1/len(a))

def weighted_average(a, weights):  
    
    """Calculates the weighted average of the input values

    Parameters
    ----------
    a : array
        An array with the values from which its weighted average is to be calculated
    weights : array
        An array whit the weights_

    Returns
    -------
    real:
        The weighted average of the values of a
    """
    s = 0
    for i in range(0,len(a)):
        s = s + a[i]*weights[i]
    return s/np.sum(weights)    