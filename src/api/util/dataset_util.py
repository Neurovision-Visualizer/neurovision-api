import pandas as pd
import os


def initialize_dataset(dataset_name):
    """
    Method for creating a dataframe for the respective dataset

    Params:
        <str> dataset_name: The name of the desired dataset

    Returns:
        <pandas.core.frame.DataFrame> A dataframe of the dataset
    """
    # Initialize key variables
    base_path = "../datasets"
    dataset_locations = {
        "heart_disease": "Heart Disease -  Binary Classification/Heart Disease - Training.csv",
        "house_price": "House Price Prediction - Regression/House Price Prediction - Training.csv",
        "iris": "Iris Prediction - Multi Classification/Iris Prediction - Training.csv"
    }
    
    # Create path for respective dataset
    dataset_path = os.path.join(base_path, dataset_locations.get(dataset_name))

    # Create dataframe
    dataframe = pd.read_csv(dataset_path)

    return dataframe

def initialize_from_json(data):

    #Create DataFrame from Dataset Received from Request
    dataframe = pd.DataFrame.from_records(data["rows"], columns=data["headings"], index=data["index"])

    return dataframe