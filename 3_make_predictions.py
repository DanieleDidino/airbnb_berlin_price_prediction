from pathlib import Path
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
import pickle

# TransformerMixin: add method ".fit_transform()"
# BaseEstimator: add methods ".get_params()" and ".set_params()"
# We need 3 methods:
# 1) .fit()
# 2) .transform()
# 3) .fit_transform() (provided by "TransformerMixin")
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    # avoid "*args" or "**kargs" in "__init__"
    def __init__(self):
        pass

    # fit is needed later for the pipilene
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        # tranformen data frame
        X_tran = pd.DataFrame()

        # Host Since
        date = pd.to_datetime(X["Host Since"], format="%Y-%m-%d")
        X_tran["year"] = date.dt.year

        # Is Superhost
        X_tran["Is Superhost"] = X["Is Superhost"]

        # Property Type
        X_tran["Property Type"] = X["Property Type"]
        X_tran.loc[X["Property Type"] == "*", "Property Type"] = np.nan

        # Room Type
        X_tran["Room Type"] = X["Room Type"]

        # Accomodates
        X_tran["Accomodates"] = X["Accomodates"]
        X_tran.loc[X_tran.Accomodates == "*", "Accomodates"] = np.nan
        X_tran["Accomodates"] = X_tran.Accomodates.astype("float")

        # Bathrooms
        X_tran["Bathrooms"] = X["Bathrooms"]
        X_tran.loc[X_tran.Bathrooms == "*", "Bathrooms"] = np.nan
        X_tran["Bathrooms"] = X_tran.Bathrooms.astype("float")

        # Bedrooms
        X_tran["Bedrooms"] = X["Bedrooms"]
        X_tran.loc[X_tran.Bedrooms == "*", "Bedrooms"] = np.nan
        X_tran["Bedrooms"] = X_tran.Bedrooms.astype("float")

        # Beds
        X_tran["Beds"] = X["Beds"]
        X_tran.loc[X_tran.Beds == "*", "Beds"] = np.nan
        X_tran["Beds"] = X_tran.Beds.astype("float")

        # Min Nights
        X_tran["Min Nights"] = X["Min Nights"]
        X_tran.loc[X_tran["Min Nights"] == "*", "Min Nights"] = np.nan
        X_tran["Min Nights"] = X_tran["Min Nights"].astype("float")

        # Instant Bookable
        X_tran["Instant Bookable"] = X["Instant Bookable"]

        return X_tran


def load_pickles(model_pickle_path):
    # open a file where the model is stored
    file = open(model_pickle_path,"rb")
    # Load the model
    model = pickle.load(file)
    # close the file
    file.close()

    return model


def make_predictions(df, model):
    prediction = model.predict(df)
    return prediction



def generate_predictions():
    test_df = pd.read_csv("data/single_row_test.csv")
    model_pickle_path = "models/final_model.pkl"

    model = load_pickles(model_pickle_path)

    # processed_df = pre_process_data(test_df, label_encoder_dict)
    prediction = make_predictions(test_df, model)
    print(prediction)


if __name__ == '__main__':
    generate_predictions()