import sys # Used to access system-specific parameters and functions (like traceback info during exceptions).
import os # Helps in dealing with file paths (like os.path.join for building cross-platform paths).
import pandas as pd
from src.exception import CustomException
from src.utils import load_object # importing a utility function to deserialize (load) saved models or transformers using pickle.

#this class is responsible for loading the pre-trained model and transformer
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features): # accepts a feature object to be used as input for prediction
        try:
            # constructs the path to the saved model and preprocessor objects from the artifacts directory
            # model.pkl pickled the trained model
            model_path=os.path.join("artifacts","model.pkl") 
            # preprocessor.pkl pickled transformer ( pipeline that preprocesses input)
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            print("Before Loading")
            # Loading both model and preprocessor using util function
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")

            # applies same transformation used during training on the input data
            data_scaled=preprocessor.transform(features)
            # predicts the outcome using the transformed features
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


# This class structure user input into the format expected by the model
class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        # Stores all passed parameters as instance variables
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

# converts the input attributes into a pandas Data Frame 
    def get_data_as_data_frame(self):
        try:

            # Creates a dictionary of features, with each value wrapped in a list to make it compatible with pd.DataFrame
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

