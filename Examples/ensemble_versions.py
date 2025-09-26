from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from desReg.des.DESRegression import DESRegression
from desReg.utils import measures as em

regressors_list = [
    linear_model.Lasso(), 
    linear_model.Ridge(), 
    KNeighborsRegressor(), 
    SVR(), 
    DecisionTreeRegressor()
]

# Baseline ensemble
des_baseline_ensemble = DESRegression(
    regressors_list=regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.all_errors,
    include_instance_hardness=False
)


## DES Ensembles
des_ensemble_ih = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=True,
    include_hardness_measures=False,
    hardness_measures_list=None,
    meta_feature='instance_hardness',
    ensemble_type='DES'
)

des_ensemble_inverse_ih = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=True,
    include_hardness_measures=False,
    hardness_measures_list=None,
    meta_feature='inverse_instance_hardness',
    ensemble_type='DES'
)

ensemble_hm = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['S2'],
    meta_feature='S2',
)

des_ensemble_s1 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    # hardness_measures_list=['S1','S2','S3', 'D', 'L1', 'EZR', 'EDS'],
    hardness_measures_list=['S1'],
    meta_feature='S1',
    ensemble_type='DES'
)

des_ensemble_inverse_s1 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    # hardness_measures_list=['S1','S2','S3', 'D', 'L1', 'EZR', 'EDS'],
    hardness_measures_list=['S1'],
    meta_feature='inverse_S1',
    ensemble_type='DES'
)

des_ensemble_s2 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    # hardness_measures_list=['S1','S2','S3', 'D', 'L1', 'EZR', 'EDS'],
    hardness_measures_list=['S2'],
    meta_feature='S2',
    ensemble_type='DES'
)

des_ensemble_inverse_s2 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    # hardness_measures_list=['S1','S2','S3', 'D', 'L1', 'EZR', 'EDS'],
    hardness_measures_list=['S2'],
    meta_feature='inverse_S2',
    ensemble_type='DES'
)
des_ensemble_s3 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    # hardness_measures_list=['S1','S2','S3', 'D', 'L1', 'EZR', 'EDS'],
    hardness_measures_list=['S3'],
    meta_feature='S3',
    ensemble_type='DES'
)

des_ensemble_inverse_s3 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    # hardness_measures_list=['S1','S2','S3', 'D', 'L1', 'EZR', 'EDS'],
    hardness_measures_list=['S3'],
    meta_feature='inverse_S3',
    ensemble_type='DES'
)

des_ensemble_d = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    # hardness_measures_list=['S1','S2','S3', 'D', 'L1', 'EZR', 'EDS'],
    hardness_measures_list=['D'],
    meta_feature='D',
    ensemble_type='DES'
)

des_ensemble_inverse_d = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    # hardness_measures_list=['S1','S2','S3', 'D', 'L1', 'EZR', 'EDS'],
    hardness_measures_list=['D'],
    meta_feature='inverse_D',
    ensemble_type='DES'
)

des_ensemble_l1 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    # hardness_measures_list=['S1','S2','S3', 'D', 'L1', 'EZR', 'EDS'],
    hardness_measures_list=['L1'],
    meta_feature='L1',
    ensemble_type='DES'
)

des_ensemble_inverse_l1 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    # hardness_measures_list=['S1','S2','S3', 'D', 'L1', 'EZR', 'EDS'],
    hardness_measures_list=['L1'],
    meta_feature='inverse_L1',
    ensemble_type='DES'
)

des_ensemble_eds = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['EDS'],
    meta_feature='EDS',
    ensemble_type='DES'
)

des_ensemble_inverse_eds = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['EDS'],
    meta_feature='inverse_EDS',
    ensemble_type='DES'
)

des_ensemble_ezr = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['EZR'],
    meta_feature='EZR',
    ensemble_type='DES'
)

des_ensemble_inverse_ezr = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['EZR'],
    meta_feature='inverse_EZR',
    ensemble_type='DES'
)

## DRS Ensembles
drs_ensemble_ih = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=True,
    include_hardness_measures=False,
    hardness_measures_list=None,
    meta_feature='instance_hardness',
    ensemble_type='DRS'
)

## DRS ensembles
drs_baseline_ensemble = DESRegression(
    regressors_list=regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.all_errors,
    include_instance_hardness=False,
    ensemble_type='DRS'
)

drs_ensemble_s1 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['S1'],
    meta_feature='S1',
    ensemble_type='DRS'
)

drs_ensemble_s2 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['S2'],
    meta_feature='S2',
    ensemble_type='DRS'
)

drs_ensemble_inverse_s2 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['S2'],
    meta_feature='inverse_S2',
    ensemble_type='DRS'
)

drs_ensemble_s3 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['S3'],
    meta_feature='S3',
    ensemble_type='DRS'
)

drs_ensemble_d = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['D'],
    meta_feature='D',
    ensemble_type='DRS'
)

drs_ensemble_l1 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['L1'],
    meta_feature='L1',
    ensemble_type='DRS'
)

drs_ensemble_eds = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['EDS'],
    meta_feature='EDS',
    ensemble_type='DRS'
)

drs_ensemble_ezr = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['EZR'],
    meta_feature='EZR',
    ensemble_type='DRS'
)

## SE Ensembles
se_baseline_ensemble = DESRegression(
    regressors_list=regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.all_errors,
    include_instance_hardness=False,
    ensemble_type='SE'
)

## DES Challenger Experiments
des_challenger_experiments = {
    'des_ensemble_ih': {
        "model": des_ensemble_ih,
    },
    'des_ensemble_inverse_ih': {
        "model": des_ensemble_inverse_ih,
    },
    'des_ensemble_s1': {
        "model": des_ensemble_s1,
    },
    'des_ensemble_inverse_s1': {
        "model": des_ensemble_inverse_s1,
    },
    'des_ensemble_s2': {
        "model": des_ensemble_s2,
    },
    'des_ensemble_inverse_s2': {
        "model": des_ensemble_inverse_s2,
    },
    'des_ensemble_s3': {
        "model": des_ensemble_s3,
    },
    'des_ensemble_inverse_s3': {
        "model": des_ensemble_inverse_s3,
    },
    'des_ensemble_d': {
        "model": des_ensemble_d,
    },
    'des_ensemble_inverse_d': {
        "model": des_ensemble_inverse_d,
    },
    'des_ensemble_l1': {
        "model": des_ensemble_l1,
    },
    'des_ensemble_inverse_l1': {
        "model": des_ensemble_inverse_l1,
    },
    'des_ensemble_eds': {
        "model": des_ensemble_eds,
    },
    'des_ensemble_inverse_eds': {
        "model": des_ensemble_inverse_eds,
    },
    'des_ensemble_ezr': {
        "model": des_ensemble_ezr,
    },
    'des_ensemble_inverse_ezr': {
        "model": des_ensemble_inverse_ezr,
    },
}

des_inverse_challenger_experiments = {
    'des_ensemble_inverse_ih': {
        "model": des_ensemble_inverse_ih,
    },
    'des_ensemble_inverse_s1': {
        "model": des_ensemble_inverse_s1,
    },
    'des_ensemble_inverse_s2': {
        "model": des_ensemble_inverse_s2,
    },
    'des_ensemble_inverse_s3': {
        "model": des_ensemble_inverse_s3,
    },
    'des_ensemble_inverse_d': {
        "model": des_ensemble_inverse_d,
    },
    'des_ensemble_inverse_l1': {
        "model": des_ensemble_inverse_l1,
    },
    'des_ensemble_inverse_eds': {
        "model": des_ensemble_inverse_eds,
    },
    'des_ensemble_inverse_ezr': {
        "model": des_ensemble_inverse_ezr,
    },
}

## DRS Challenger Experiments
drs_challenger_experiments = {
    'drs_ensemble_ih': {
        "model": drs_ensemble_ih,
    },
    'drs_ensemble_s1': {
        "model": drs_ensemble_s1,
    },
    'drs_ensemble_s2': {
        "model": drs_ensemble_s2,
    },
    'drs_ensemble_s3': {
        "model": drs_ensemble_s3,
    },
    'drs_ensemble_d': {
        "model": drs_ensemble_d,
    },
    'drs_ensemble_l1': {
        "model": drs_ensemble_l1,
    },
    'drs_ensemble_eds': {
        "model": drs_ensemble_eds,
    },
    'drs_ensemble_ezr': {
        "model": drs_ensemble_ezr,
    },
}

## SE Ensembles
se_ensemble_ih = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.all_errors,
    include_instance_hardness=True,
    include_hardness_measures=False,
    hardness_measures_list=None,
    meta_feature='instance_hardness',
    ensemble_type='SE'
)

se_ensemble_s1 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['S1'],
    meta_feature='S1',
    ensemble_type='SE'
)

se_ensemble_s2 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['S2'],
    meta_feature='S2',
    ensemble_type='SE'
)

se_ensemble_s3 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['S3'],
    meta_feature='S3',
    ensemble_type='SE'
)

se_ensemble_d = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['D'],
    meta_feature='D',
    ensemble_type='SE'
)

se_ensemble_l1 = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['L1'],
    meta_feature='L1',
    ensemble_type='SE'
)

se_ensemble_eds = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['EDS'],
    meta_feature='EDS',
    ensemble_type='SE'
)

se_ensemble_ezr = DESRegression(
    regressors_list = regressors_list,
    random_state=0,
    n_estimators_bag=20,
    competence_region='knn',
    k=5,
    competence_level=em.sum_absolute_error_weighted,
    include_instance_hardness=False,
    include_hardness_measures=True,
    hardness_measures_list=['EZR'],
    meta_feature='EZR',
    ensemble_type='SE'
)

se_challenger_experiments = {
    'se_ensemble_ih': {
        "model": se_ensemble_ih,
    },
    "se_baseline_ensemble": {
        "model": se_baseline_ensemble,
    }
    # 'se_ensemble_s1': {
    #     "model": se_ensemble_s1,
    # },
    # 'se_ensemble_s2': {
    #     "model": se_ensemble_s2,
    # },
    # 'se_ensemble_s3': {
    #     "model": se_ensemble_s3,
    # },
    # 'se_ensemble_d': {
    #     "model": se_ensemble_d,
    # },
    # 'se_ensemble_l1': {
    #     "model": se_ensemble_l1,
    # },
    # 'se_ensemble_eds': {
    #     "model": se_ensemble_eds,
    # },
    # 'se_ensemble_ezr': {
    #     "model": se_ensemble_ezr,
    # },
}
