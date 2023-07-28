from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import datetime
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import matplotlib.pyplot as plt

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


def load_prices():
    df_path = "data/train_airbnb_berlin.csv"
    df = pd.read_csv(df_path)

    df = df.loc[~df.Price.isna(), :]
    price = df.loc[:, "Price"].copy()

    # define range
    price_min = 20
    price_max = 150
    price = price[price < price_max]
    price = price[price > price_min]

    return price


def load_col_values():
    col_values_path = "models/col_values.pkl"
    # open a file where the model is stored
    file = open(col_values_path,"rb")
    # Load the model
    col_values = pickle.load(file)
    # close the file
    file.close()

    return col_values


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


def generate_predictions(df):
    model_pickle_path = "models/final_model.pkl"

    model = load_pickles(model_pickle_path)

    prediction = make_predictions(df, model)
    prediction = np.round(prediction)

    return prediction[0]

if __name__ == '__main__':

    prices = load_prices()
    
    col_values = load_col_values()

    # make the application
    st.title("Airbnb Berlin Price Prediction")
    st.text("Enter your requirements: ")

    # making customer requirements data input
    host_since = st.date_input("Select the Host Since Date :",
                              min_value=datetime.date(col_values["year"]["min"], 1, 1),
                              max_value=datetime.date(col_values["year"]["max"],12,31),
                              value=datetime.date(col_values["year"]["mode"], 1, 1),
                              #format="%Y-%m-%d", #"YYYY/MM/DD",
                              label_visibility="visible")
    
    if col_values["Is Superhost"]["mode"] == "f":
        is_superhost_order = ["No", "Yes"]
    else:
        is_superhost_order = ["Yes", "No"]
    is_superhost = st.selectbox('Is Superhost? :',
                                is_superhost_order)
    if is_superhost == "Yes":
        superhost = "t"
    else:
        superhost = "f"
      
    property_type = st.selectbox('Please select the property type :',
                             col_values["Property Type"]["categories"])
    
    room_type = st.selectbox('Please select the Room Type :',
                             col_values["Room Type"]["categories"])
    
    accomodates = st.slider('Please select the number of Accomodates :',
                            max_value=col_values["Accomodates"]["max"],
                            min_value=col_values["Accomodates"]["min"],
                            value=col_values["Accomodates"]["mode"])
    
    bedrooms = st.slider('Please select the number of Bedrooms :',
                         max_value=col_values["Bedrooms"]["max"],
                         min_value=col_values["Bedrooms"]["min"],
                         value=col_values["Bedrooms"]["mode"],
                         step=1.0,
                         format="%i")
    
    
    bathrooms = st.slider('Please select the number of Bathrooms :',
                          max_value=col_values["Bathrooms"]["max"],
                          min_value=col_values["Bathrooms"]["min"],
                          value=col_values["Bathrooms"]["mode"],
                          step=0.5,
                          format="%f")
    
    beds = st.slider('Please select the number of Beds :',
                     max_value=col_values["Beds"]["max"],
                     min_value=col_values["Beds"]["min"],
                     value=col_values["Beds"]["mode"],
                     step=1.0,
                     format="%i")
    

    min_nights = st.slider('Please select the minimum number of nights :',
                           max_value=col_values["Min Nights"]["max"],
                           min_value=col_values["Min Nights"]["min"],
                           value=col_values["Min Nights"]["mode"],
                           step=1.0,
                           format="%i")
    
    if col_values["Instant Bookable"]["mode"] == "f":
        instant_bookable_order = ["No", "Yes"]
    else:
        instant_bookable_order = ["Yes", "No"]
    
    instant_bookable = st.selectbox('Do you prefer Instant Booking :',
                                    instant_bookable_order)
    if instant_bookable == "Yes":
        instantbookable = "t"
    else:
        instantbookable = "f"
    
    input_dict = {'Host Since': host_since,
                  'Is Superhost': superhost,
                  'Property Type': property_type,
                  'Room Type': room_type,
                  'Accomodates': accomodates,
                  'Bedrooms': bedrooms,
                  'Bathrooms':bathrooms,
                  'Beds': beds,
                  'Min Nights': min_nights,
                  'Instant Bookable': instantbookable,
                  }
    input_data = pd.DataFrame([input_dict])

    # generate the prediction for the customer
    if st.button("Predict Airbnb Berlin Price"):
        pred = generate_predictions(input_data)
        pred = int(pred)
        churn_text=f"The predicted price per night is:"
        st.markdown(churn_text, unsafe_allow_html=True)
        churn_text = f'<p style="font-family:Helvetica; color:Red; font-size: 20px;">{pred}€</p>'
        st.markdown(churn_text, unsafe_allow_html=True)

        churn_text = "-"*80
        st.markdown(churn_text, unsafe_allow_html=True)

        less_than_price = sum(prices < pred)
        percent_less = (less_than_price / len(prices)) * 100
        percent_less = round(percent_less)
        churn_text = f"The predicted price is is higher than {percent_less}% of the available prices."
        st.markdown(churn_text, unsafe_allow_html=True)

        churn_text = "The histogram represent the distribution of the Prices"
        st.markdown(churn_text, unsafe_allow_html=True)
        churn_text = "The red line is the predicted values"
        st.markdown(churn_text, unsafe_allow_html=True)
        
        fig, ax = plt.subplots()
        ax.hist(prices)
        ax.axvline(pred, color='r', linestyle='dashed', linewidth=2)
        ax.set(ylabel=None, yticklabels=[], yticks=[])
        ax.set(xlabel="Price (in Euro)", xticks=list(range(20, 160, 10)))
        plt.show()
        st.pyplot(fig)
    
    # prices.hist()
    # plt.show()
    # st.pyplot()

    
