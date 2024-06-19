from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer


class featureEngineeringStage:
    def __init__(self, columns, selection):
        # define categorical columns
        self.columns = columns
        # define all columns
        self.selection = selection

    def transform(self, X, y=None):
        """Transform columns of X using MinMaxScaler and LabelEncoder"""
        output = X.copy()
        # define numeric columns 
        num_columns = [col for col in self.selection if col not in self.columns]

        # Apply LabelEncoder to categorical columns, OneHotEncoder does not fit the schema
        label_encoders = {}
        for col in self.columns:
            label_encoders[col] = LabelEncoder()
            output[col] = label_encoders[col].fit_transform(output[col]) 

        # Apply MinMaxScaler to numerical columns
        scaler = MinMaxScaler()
        for col in num_columns:
            output[col] = scaler.fit_transform(output[[col]]) 

        return output[self.selection]

    def fit(self, X, y=None):
        return self
    
