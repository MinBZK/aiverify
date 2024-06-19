import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from test_models.pipelineCustomClass import featureEngineeringStage

# create instance of featureEngineeringStage for transforming and fitting to align with set-up of AIverify
# define all categorical columns
columns = ["gender", "nationality", "ind-debateclub", "ind-programming_exp",
               "ind-international_exp", "ind-entrepeneur_exp",
                "ind-exact_study", "ind-degree"]
# define all columns
selection = ["gender", "nationality", "ind-debateclub", "ind-programming_exp",
               "ind-international_exp", "ind-entrepeneur_exp",
                "ind-exact_study", "ind-degree", "age", "ind-university_grade", "ind-languages"]
featureEngineeringStage = featureEngineeringStage(columns, selection)

#read in data 
df = pd.read_csv('./recruitmentdataset-2022-1.3.csv', delimiter = ',')

#partly used code from thesis: https://github.com/GuusjeJuijn/fairness-perceptions/blob/main/Recruitment%20Prediction%20-%20Model%20Development.ipyn

#data preprocessing
data = df[1000:2000] #the dataset contains information about 4 different companies, to make sure our data makes more sense we only select the data from 1 company
data.info()
data.drop('Id', axis=1, inplace=True) # drop ID column as it is not important for our analysis
data.drop('sport', axis=1, inplace=True) #drop the sport column, to decrease the number of features

# transform boolean column types, and ind-languages column, into objects
data['ind-debateclub'] = data['ind-debateclub'].astype('O')
data['ind-programming_exp'] = data['ind-programming_exp'].astype('O')
data['ind-international_exp'] = data['ind-international_exp'].astype('O')
data['ind-entrepeneur_exp'] = data['ind-entrepeneur_exp'].astype('O')
data['ind-exact_study'] = data['ind-exact_study'].astype('O')
data['decision'] = data['decision'].astype('O')

#split into train and test set
input_attributes = data.iloc[0:,0:12] # Split the data into input attributes and target attribute
target_attribute = data['decision']

cat_columns = ["gender", "nationality", "ind-debateclub", "ind-programming_exp",
               "ind-international_exp", "ind-entrepeneur_exp",
                "ind-exact_study", "ind-degree"]
num_columns = ["age", "ind-university_grade", "ind-languages"]


X = input_attributes[cat_columns + num_columns]

# split data into train and test set
x_train, x_test, y_train, y_test = train_test_split(X, target_attribute.astype('int'), test_size=0.25, random_state=42)

x_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# create a scikit-learn Pipeline object that consists of two stages:
classifier = Pipeline(
    [
        # Stage 1: "preprocess" using featureEngineeringStage
        ("preprocess", featureEngineeringStage),

        # Stage 2: "classifier" using LogisticRegression
        ("classifier", LogisticRegression(C=10,random_state=42)),
    ]
)

classifier.fit(X = x_train, y = y_train)

# serialize and save the pipeline to the specified path
folder_path = 'test_models' #define the path to the target folder
os.makedirs(folder_path, exist_ok=True) # ensure the folder exists
file_path = os.path.join(folder_path, 'pipeline.sav') # full path to the .sav file

with open(file_path, 'wb') as file: #save the testing dataset to a .sav file
    pickle.dump(classifier, file)

# serialize and save the dataset to the specified path
folder_path_2 = 'test_datasets' #define the path to the target folder
os.makedirs(folder_path_2, exist_ok=True) # ensure the folder exists
file_path_2 = os.path.join(folder_path_2, 'dataset.sav') # full path to the .sav file

with open(file_path_2, 'wb') as file: #save the dataset to a .sav file
    pickle.dump(data, file)

# create y_test data in right format 
df_y = data['decision'].to_frame().astype(int)
# serialize and save the dataset to the specified path
file_path_3 = os.path.join(folder_path_2, 'dataset_ytest.sav') # full path to the .sav file

with open(file_path_3, 'wb') as file: #save the dataset to a .sav file
    pickle.dump(df_y, file)

