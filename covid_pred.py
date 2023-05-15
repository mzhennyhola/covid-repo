import streamlit as st
from PIL import Image
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from datetime import date
import joblib
import warnings
import time
warnings.simplefilter('ignore')


def get_user_agent(country):
    url = f"https://nominatim.openstreetmap.org/search?q={country}&format=json"
    response = requests.get(url)
    response_json = response.json()
    latitude = response_json[0]["lat"]
    longitude = response_json[0]["lon"]
    return latitude, longitude

def read_and_train():
    print("reading data and preprocessing ...")
    # read data and preprocess
    pd.set_option('display.max_columns', None)
    data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    data = data.rename(columns={'Province/State': 'province', 'Country/Region': 'country', 'Lat': 'lat', 'Long': 'long'})
    data.isnull().sum().sort_values(ascending=False)
    data.drop(['province'], axis=1, inplace=True)
    data = data.dropna()

    # melt data to have fewer columns and more rows
    covid = pd.melt(data, id_vars=['country', 'lat', 'long'], var_name='date', value_name='confirmed_cases')
    covid['date'] = pd.to_datetime(covid['date'])
    covid = covid.sort_values(['country', 'date'])
    covid = covid.reset_index(drop=True)
    covid.head(3)

    # create a new column to include the year information derived from the dates in the data
    covid['year'] = covid['date'].dt.year
    covid = covid.reindex(columns=['year','country','lat','long','confirmed_cases'])
    covid.head(2)

    # categorize data in order to encode and standardize per category
    cat = covid.select_dtypes(include='object')
    num = covid.select_dtypes(include='number')
    cat.head(2)
    num.head(2)

    # split into x and y
    x = covid.drop(['confirmed_cases', 'country'], axis=1)
    y = covid['confirmed_cases']

    # split into train and test
    x_train,x_valid,y_train,y_valid = train_test_split(x,y,test_size=0.2,random_state=20)

    # create pipeline
    global pipeline_rfr
    pipeline_rfr = Pipeline([('scaler', MinMaxScaler()), ('classifier', RandomForestRegressor())])
    pipeline_dtr = Pipeline([('scaler', MinMaxScaler()), ('classifier', DecisionTreeRegressor())])
    pipeline_lr = Pipeline([('scaler', MinMaxScaler()), ('classifier', LinearRegression())])
    pipeline_gbr = Pipeline([('scaler', MinMaxScaler()), ('classifier', GradientBoostingRegressor())])

    # Create a list of the pipelines
    pipelines = [pipeline_rfr, pipeline_dtr, pipeline_lr, pipeline_gbr]

    # Create a dictionary of Pipelines for ease of reference 
    pipeline_dict = {0: 'Random Forest', 1: 'Decision Tree', 2: 'Linear Regression', 3: 'Gradient Boosting'}
    
    global best_pipeline
    best_accuracy = 0.0
    best_classifier = 0
    best_pipeline = ""

    print("starting pipeline fitting ...")
    start_time = time.time()

    for pipe in pipelines:
        pipe.fit(x_train, y_train)

    print(f"finished pipeline fitting in {time.time() - start_time} seconds")

    for i, model in enumerate(pipelines):
        print(f'\n{pipeline_dict[i]} Training Accuracy: {model.score(x_train, y_train)}')

    for i, model in enumerate(pipelines):
        print(f'\n{pipeline_dict[i]} Test Accuracy: {model.score(x_valid, y_valid)}')

    for i, model in enumerate(pipelines):
        if model.score(x_valid, y_valid) > best_accuracy:
            best_accuracy = model.score(x_valid, y_valid)
            best_pipeline = model
            best_classifier = i
    print(f'Classifier with the best accuracy: {pipeline_dict[best_classifier]}')
    joblib.dump(pipeline_rfr, 'covid_pred.pkl')

# read_and_train()

# --------------- Streamlit deployment ------------------ #

# def run_streamlit():
#   st.title('Covid-19 Confirmed Cases Predictor')
#   st.markdown('This app predicts the value of covid-19 confirmed cases based on a trend analysis of data from inception of the pandemic to 2023')
#   st.image('covid.jpg')
#   st.write('COVID-19 is a highly infectious respiratory illness caused by the novel coronavirus SARS-CoV-2. It was first identified in Wuhan, China in December 2019 and has since spread to become a global pandemic. The virus is primarily spread through respiratory droplets when an infected person talks, coughs, or sneezes, and can also be transmitted by touching a surface contaminated with the virus and then touching one\'s face.\n Symptoms of COVID-19 can range from mild to severe and include fever, cough, shortness of breath, fatigue, loss of taste or smell, and body aches. Some people may experience no symptoms at all. \n COVID-19 can lead to severe respiratory illness, hospitalization, and death, especially in people with underlying health conditions or those over the age of 60. Vaccines are now available and effective in preventing severe illness and hospitalization.')
#   username = st.text_input('What is your name?')
#   button = st.button('Please click me to submit.')
#   if button:
#     if username != '':
#       st.markdown(f'Hello, {username}!')
#     else:
#       st.warning('Please input your username to continue.')

#   st.sidebar.write('Prediction Metrics')

#   year = st.sidebar.selectbox('What year do you want to predict?', (2020, 2021, 2022, 2023, 2024, 2025))

#   # Initialize geocoding API
#   country = st.sidebar.text_input('Enter the name of your desired country: ')
#   longitude, latitude = 0, 0
  
#   if country != '':
#     latitude, longitude = get_user_agent(country)

#     long = st.sidebar.write(f'The longitude of your location is: {longitude}')

#     lat = st.sidebar.write(f'The latitude of your location is: {latitude}')

#   predict_button = st.sidebar.button('Predict the number of confirmed cases')

#   # user input
#   input_variables = [[year, longitude, latitude]]
#   input_v = np.array(input_variables)

#   frame = ({'year':[year], 'longitude': [longitude], 'latitude': [latitude]})
#   st.write('These are your input variables: ')
#   frame = pd.DataFrame(frame)
#   frame = frame.rename(index = {0: 'Value'})
#   frame = frame.transpose()
#   st.write(frame)

#   # load the model
#   model_ = joblib.load(open('covid_predictor.pkl','rb'))
#   regressor = model_.predict(input_v)
#   current_date = date.today()

#   if predict_button:
#     st.write(f'The estimated average number of confirmed cases in {country} in the year {year} is, {int(regressor[0])}')

# run_streamlit()

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from datetime import date

st.set_page_config(page_title="Covid-19 Confirmed Cases Predictor", page_icon=":guardsman:", layout="wide")

st.title('Covid-19 Confirmed Cases Predictor')
st.markdown('This app predicts the value of covid-19 confirmed cases based on a trend analysis of data from inception of the pandemic to 2023')
st.image('covid.png')
st.write('COVID-19 is a highly infectious respiratory illness caused by the novel coronavirus SARS-CoV-2. It was first identified in Wuhan, China in December 2019 and has since spread to become a global pandemic. The virus is primarily spread through respiratory droplets when an infected person talks, coughs, or sneezes, and can also be transmitted by touching a surface contaminated with the virus and then touching one\'s face.\n Symptoms of COVID-19 can range from mild to severe and include fever, cough, shortness of breath, fatigue, loss of taste or smell, and body aches. Some people may experience no symptoms at all. \n COVID-19 can lead to severe respiratory illness, hospitalization, and death, especially in people with underlying health conditions or those over the age of 60. Vaccines are now available and effective in preventing severe illness and hospitalization.')

username = st.text_input('What is your name?')
button = st.button('Please click me to submit.')
if button:
    if username != '':
        st.markdown(f'Hello, {username}!')
    else:
        st.warning('Please input your username to continue.')

st.sidebar.write('Prediction Metrics')

year = st.sidebar.selectbox('What year do you want to predict?', (2020, 2021, 2022, 2023, 2024, 2025))

# Initialize geocoding API
country = st.sidebar.text_input('Enter the name of your desired country: ')
longitude, latitude = 0, 0

if country != '':
    url = f"https://nominatim.openstreetmap.org/search?q={country}&format=json"
    response = requests.get(url).json()
    latitude = float(response[0]['lat'])
    longitude = float(response[0]['lon'])

    long = st.sidebar.write(f'The longitude of your location is: {longitude}')

    lat = st.sidebar.write(f'The latitude of your location is: {latitude}')

predict_button = st.sidebar.button('Predict the number of confirmed cases')

# Load the pre-trained model
model = joblib.load(open('covid_pred.pkl', 'rb'))

# User input
input_variables = [[year, longitude, latitude]]
input_v = np.array(input_variables)

frame = ({'year': [year], 'longitude': [longitude], 'latitude': [latitude]})
st.write('These are your input variables: ')
frame = pd.DataFrame(frame)
frame = frame.rename(index={0: 'Value'})
frame = frame.transpose()
st.write(frame)

# Make prediction
if predict_button:
    regressor = model.predict(input_v)
    current_date = date.today()
    st.write(f'The estimated average number of confirmed cases in {country} in the year {year} is, {int(regressor[0])}')