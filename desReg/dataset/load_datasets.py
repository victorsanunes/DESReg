import pkg_resources
import pandas as pd

from ucimlrepo import fetch_ucirepo
DATASETS_PATH = "Datasets/external_datasets"


def load_Student_Mark():
    """Return the dataset Student Mark.
      The data consists of Marks of students including their study time & number of courses.
      The dataset is downloaded from UCI Machine Learning Repository.

    Number of Instances: 100
    Number of Attributes: 3 including the target variable.

    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, "Student_Marks.csv")
    return pd.read_csv(stream)


def load_ccpp():
    tabs = pd.ExcelFile(f"{DATASETS_PATH}/CCPP/Folds5x2_pp.xlsx").sheet_names
    all_tabs_data = []
    partition_number = 0
    for tab in tabs:
        tab_data = pd.read_excel(
            f"{DATASETS_PATH}/CCPP/Folds5x2_pp.xlsx", sheet_name=tab
        )
        tab_data["partition"] = partition_number
        partition_number += 1
        all_tabs_data.append(tab_data)
    raw_df = pd.concat(all_tabs_data, ignore_index=True)
    dataset_metadata = {
        "target_column": "PE",
        "features": [
            "AT",
            "AP",
            "AH",
            "AFDP",
            "GTEP",
        ],
        "data": raw_df,
    }
    return dataset_metadata


def load_airfoil():
    raw_df = pd.read_csv(
        f"{DATASETS_PATH}/airfoil/airfoil_self_noise.dat", sep="\s+", header=None
    )
    raw_df.columns = [
        "frequency",
        "angle",
        "chord_length",
        "free_stream_velocity",
        "suction_side_displacement_thickness",
        "sound_pressure",
    ]
    dataset_metadata = {
        "target_column": "sound_pressure",
        "features": [
            "frequency",
            "angle",
            "chord_length",
            "free_stream_velocity",
            "suction_side_displacement_thickness",
        ],
        "data": raw_df,
    }
    return dataset_metadata

def load_air_quality():
    # Load Excel file for air quality dataset
    air_quality_df = pd.read_excel(f"{DATASETS_PATH}/air-quality/AirQualityUCI.xlsx")
    metadata = {
        "target_column": "CO(GT)",
        "features": [
            "CO(GT)",
            "PT08.S1(CO)",
            "NMHC(GT)",
            "C6H6(GT)",
            "PT08.S2(NMHC)",
            "NOx(GT)",
            "PT08.S3(NOx)",
            "NO2(GT)",
            "PT08.S4(NO2)",
            "PT08.S5(O3)",
            "T",
            "RH",
            "AH",
        ],
        "data": air_quality_df,
    }
    return metadata
