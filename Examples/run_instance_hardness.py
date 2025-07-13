import pandas as pd
import numpy as np

from desReg.utils import instance_hardness as ih
from run_experiments import partitions_dict


train_df_dict= {}
datasets = {}

def load_datasets():
    for partition_alias, partition_name in partitions_dict.items():
        n_fold = 5
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for p in range(1,n_fold+1):
            name_train = partition_name+str(p)+'tra.dat'
            name_test = partition_name+str(p)+'tst.dat'

            data_train = pd.read_csv(name_train,header = None)
            columns = [f"V{i}" for i in range(data_train.shape[1] - 1)]
            columns.append("y")
            data_train.columns = columns
            X_train = data_train.iloc[:,:-1].to_numpy()
            y_train = np.ravel(data_train.iloc[:, -1:])

            data_test = pd.read_csv(name_test, header=None)
            data_test.columns = columns
            X_test = data_test.iloc[:,:-1].to_numpy()
            y_test = np.ravel(data_test.iloc[:, -1:])

            train_df = pd.concat([train_df, data_train], ignore_index=True)
            test_df = pd.concat([test_df, data_test], ignore_index=True)

        datasets[partition_alias] = dict(
            train_df = train_df,
            test_df = test_df
        )
    return datasets


if __name__ == "__main__":
    datasets = load_datasets()
    all_measures_dict = {}
    for name, data in datasets.items():
        if name in ["yacht"]:
            all_measures_dict[name] = dict()
            X_train = data['train_df'].drop(columns=['y'])
            y_train = data['train_df'][['y']]
            all_measures = ih.get_instance_hardness(X_train, y_train)
            all_hardness_measures = ih.get_hardness_measures(X_train, y_train)
            all_measures_dict[name]['instance_hardness'] = all_measures
            all_measures_dict[name]['hardness_measures'] = all_hardness_measures
            breakpoint()
