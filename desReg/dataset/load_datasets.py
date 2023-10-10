import pkg_resources
import pandas as pd


def load_Student_Mark():
    """Return the dataset Student Mark.
      The data consists of Marks of students including their study time & number of courses. 
      The dataset is downloaded from UCI Machine Learning Repository.

    Number of Instances: 100
    Number of Attributes: 3 including the target variable.

    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'Student_Marks.csv')
    return pd.read_csv(stream)