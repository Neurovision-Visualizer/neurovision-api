from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class DataHandler(object):

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.dataframe2 = dataframe
    
    def getDataFrame(self):
        return self.dataframe
    
    def removeFeature(self, featureName):
        self.dataframe.drop(featureName, inplace=True, axis=1)
    
    def removeInvalidData(self):
        self.dataframe.fillna("")
        self.dataframe.replace('', np.nan, inplace=True)
        self.dataframe.dropna(inplace=True)

    def translateData(self):
        # train.day = train.day.astype('category')
        mydataframe = self.dataframe
        data = mydataframe.dtypes[mydataframe.dtypes == np.object]
        columnNames = list(data.index)
        for i in columnNames:
            mydataframe[i] = mydataframe[i].astype('category')
            catagories = mydataframe[i].unique()
            for cat_index in range(len(catagories)):
                mydataframe[i] = mydataframe[i].replace([catagories[cat_index]], cat_index + 1)
        self.dataframe = mydataframe

    def normalizeData(self):
        scaler = preprocessing.MinMaxScaler()
        names = self.dataframe.columns
        d = scaler.fit_transform(self.dataframe)
        normalize_df = pd.DataFrame(d, columns = names)
        self.dataframe = normalize_df
    
    
    def denormalizeData(self):
        pass

    def setInputs(self):
        n = len(self.dataframe.columns) - 1
        self.inputs  = self.dataframe.iloc[: , :n]
    
    def setOutput(self):
        self.output = self.dataframe.iloc[: , -1:]
    
    def dataset_split(self, train):
        test_size = 1 - (train / 100)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.inputs, self.output, random_state=100, test_size = test_size)

    def toJSON(self, choice=0):
        if choice == 0:
            # return self.dataframe.iloc[1:10, ].to_dict(orient="split")
            return self.dataframe.to_dict(orient="split")
        elif choice == 1:
            # return self.dataframe2.iloc[1:10, ].to_dict(orient="split")
            return self.dataframe2.to_dict(orient="split")

    

# from controllers.DataHandler import DataHandler
# import pandas as pd
# data = pd.read_csv(r"C:\Users\Jantae Leckie\Desktop\Computing\3rd Year\COMP3901\Code\neurovision\datasets\Heart Disease -  Binary Classification\Heart Disease - Training.csv")
# dataframe = pd.DataFrame(data)
# datahandle = DataHandler(dataframe) 
# datahandle.getDataFrame()