import os #file/path manupulation
import sys # help catch complete stack trace when an exception is raised

import numpy as np 
import pandas as pd
import dill # to serialize (save) python objects
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)  # get the folder path 
 
        os.makedirs(dir_path, exist_ok=True)  # create folder 

        with open(file_path, "wb") as file_obj: # open folder in binary write mode
            pickle.dump(obj, file_obj) # serialize(save/pickle) the object into that file

    except Exception as e:
        raise CustomException(e, sys)

# takes train/test data, models, params and evaluate them using GRIDSEARCH CV    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {} # dictionary to store test R^2 scores for each model

        # loop through the models and get their corresponding hyperparameters
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            
            # Runs grid search cross-validation for each model with the given param grid.
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            # Set the model's parameters to the best found, and retrain on the full training set.
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # Get predictions on both sets.

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Compute R² 

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # saves only test set r^2 for each model

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

# This function loads back a pickled object (e.g., a trained model or preprocessor).   
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:  # Opens the file in read-binary mode
            return pickle.load(file_obj)  # Returns the deserialized Python object by unpickling the object back to memory

    except Exception as e:
        raise CustomException(e, sys)
