import sys #allows interaction with the python runtime system
from dataclasses import dataclass #automatically add special method like __init__()

import numpy as np #DS library
import pandas as pd # DS library
from sklearn.compose import ColumnTransformer # Applies different preprocessing pipelines to different columns
from sklearn.impute import SimpleImputer # Fills in missing values (eg. mode or median)
from sklearn.pipeline import Pipeline # chains preprocessing steps into a single object
from sklearn.preprocessing import OneHotEncoder,StandardScaler # encode categorical data and standardize numerical data

from src.exception import CustomException # custom handler error class
from src.logger import logging # custom logger
import os

from src.utils import save_object # save python objects with pickle 

@dataclass # auto-generate init methods
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl") #Hold a single config path to save the preprocess obj as pkl file (artifacts/preprocessor.pkl)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig() #initialize class using datatransformationconfig

# this method set up all preprocessing logic
    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"] # define which columns are numerical and categorical
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Fill missing value and apply standardization

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            # fill missing values, apply one-hot encoding and apply standardization

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Apply num pipeline and cat pipeline to respective columns

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    # accepts paths to train and test csv files   
    def initiate_data_transformation(self,train_path,test_path):

        try:
            #load datasets
            train_df=pd.read_csv(train_path) 
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Get the columntransformer created earlier

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Splits input features and target variable for training data.

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            # applies transformation (fit on training, transform on test)
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # Concatenates features and target into a single NumPy array (like [X|y])
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            
            # pickles the transformer object for future use
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            # returns transformed data and the path to the saved preprocessor
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
