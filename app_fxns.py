import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from scipy.stats import poisson
import scipy as sc
import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import urllib
import sys

import multiprocessing as mp
from multiprocessing import Pool

import model_fxns as fxns

pd.set_option('display.max_columns', None)

statepops = pd.read_csv('DataUpdate/data/StatePops.csv')

url = 'https://raw.githubusercontent.com/klocey/StateCovidData/main/data/COVID-CASES-DF.txt'
locs_df = pd.read_csv(url, sep='\t')
            

locs_df = locs_df[locs_df['Country/Region'] == 'US']
locs_df = locs_df[~locs_df['Province/State'].isin(['US', 'American Samoa', 'Northern Mariana Islands',
                                                'Diamond Princess', 'Grand Princess', 'Recovered', 
                                                 'United States Virgin Islands', 'Virgin Islands, U.S.',
                                                'Wuhan Evacuee'])]

locs_df.drop(columns=['Unnamed: 0'], inplace=True)
locations = sorted(list(set(locs_df['Province/State'])))
locations.insert(0, locations.pop(locations.index('Illinois')))
locs_df = 0

counties_df = []
with open('DataUpdate/data/States_Counties.txt', 'rb') as csvfile:
                counties_df = pd.read_csv(csvfile, sep='\t')
try:
    counties_df.drop(['Unnamed: 0'], axis=1, inplace=True)
except:
    pass

counties = list(set(counties_df['Admin2']))
counties.append('Entire state or territory')
counties_df = 0

models = ['Time series analysis', 'Logistic (multi-phase)', 'Gaussian (multi-phase)', 'Phase Wave', 'Quadratic', 'Exponential']

day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
            'Friday', 'Saturday','Sunday']






    

def description_card1():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card1",
        children=[
            html.H5("Center for Quality, Safety and Value Analytics", style={
            'textAlign': 'left',
            #'color': '#2F9314'
        }),
            html.Div(
                id="intro1",
                children="Obtain forecasts for COVID-19 cases using a suite of models, and forecasts for your patient census, discharges, and PPE needs using customizable variables",
            ),
            html.Br(),
        ],
    )


def description_card1b():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card1b",
        children=[
            html.H5("Center for Quality, Safety and Value Analytics", style={
            'textAlign': 'left',
            #'color': '#2F9314'
        }),
            html.Div(
                id="intro1b",
                children="Obtain forecasts for active COVID cases among your employees. THIS TAB IS UNDER CURRENT DEVELOPMENT. ",
            ),
            html.Br(),
        ],
    )


def description_card2():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card2",
        children=[
            html.H5("Contact Information", style={
            'textAlign': 'left',
            }),
            dcc.Markdown("-------"),
            dcc.Markdown("**App Development & Modeling**"),
            dcc.Markdown("Kenneth J. Locey, Ph.D., Data Science Analyst, email: kenneth_j_locey@rush.edu"),
            dcc.Markdown("**Center for Quality, Safety and Value Analytics: Leadership**"),
            dcc.Markdown("Thomas A. Webb, MBA, Associate Vice President, email: Thomas_A_Webb@rush.edu"),
            dcc.Markdown("Bala N. Hota, MD, MPH, Vice President, Chief Analytics Officer, email: Bala_Hota@rush.edu"),
            html.Br(),
        ],
    )


def description_card3():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card3",
        children=[
            html.H5("Important Updates", style={
            'textAlign': 'left',
            }),
            dcc.Markdown("-------"),
            dcc.Markdown("We have added two new models. These are the logistic (multiphase) and Gaussian (multiphase)." +
                         " We have removed the 2-phase logistic and 2-phase sine logistic because the new models offer a faster" +
                         " and more elegant solution. We also removed the SEIR-SD model because its poor performance did not justify" +
                         " the computational resources to maintain it. We have also added functionality to forecast active cases." +
                         " and to allow users to select county level data."),
            
            html.Br(),
            html.H5("Instructions for using the COVID Calculator", style={
            'textAlign': 'left',
            }),
            dcc.Markdown("-------"),
            dcc.Markdown("**MODEL FORECASTS**"),
            dcc.Markdown("**1. Select a State and a Model to fit.**" +
                         " Choose from 4 models to obtain forecasts for COVID-19 cases across US states and select terroritories." + 
                         " Some models run very quickly (exponential, logistic, quadratic, gaussian)." + 
                         " Other models take several seconds to run (2-phase sine-logistic, 2-phase logistic)."
                         ),
                         
            dcc.Markdown("**2. Fitting models to data.**" +
                         " The model you choose is automatically fitted to data on cumulative cases that have been observed for the state you have chosen." +
                         " This fitting is conducted by using numerical optimization on model parameters." +
                         " Coefficients of determination (r-square values) represent the portion of variation in cumulative cases that is explained by the model." +
                         " An r-square of 1.0 means that the prediction captures 100% of variation in the observed data." + 
                         " However, these r-square values do not capture how well a chosen model is performing in the most recent days."
                         ),
                         
            dcc.Markdown("**3. Interpreting what's plotted.**" + 
                         " The Model Forecast box graphs the observed data (black dots) and model forecasts (colored lines) across several days." +
                         " Variation in these forecasts, from the current day to a 10-day old forecast, can help you determine which model is providing " +
                         "the most accurate predictions for your state."
                         
                         ),
            dcc.Markdown("**4. Downloading csv file.**" + 
                         " Click on the \"Download CSV\" text to download a csv file of the plotted data."
                         ),
                         
            
            html.Br(),
            dcc.Markdown("**FORECASTED PATIENT CENSUS AND FORECASTED PATIENT DISCHARGES**"),
            dcc.Markdown("**1. Select Hospital Variables.**" + 
                         " Use the slider buttons to select the values for numerous variables that influence the forecasting of your expected patient census and discharges." +
                         " These forecasts are automatically generated and are based on forecasts of the modeling above, numerous hospital variables (under Select Hospital Variables), " +
                         "modeling of daily patient turnover, and modeling of time lags in patient visits."
                         ),
            
            dcc.Markdown("**2. Downloading csv file.**" + 
                         " Click on the \"Download CSV\" text below the Patient Census and Discharge Table" + 
                         " to download a csv file of the tabulated data."
                         ),
            
            html.Br(),
            dcc.Markdown("**FORECASTED PERSONAL PROTECTIVE EQUIPMENT (PPE) SUPPLY NEEDS**"),
            dcc.Markdown("**1. Select Daily PPE Needs.**" +
                         "Use the slider buttons to select the values for daily per patient PPE needs." + 
                         "Forecasts for PPE supply needs are based on the combination of the patient census forecast and " +
                         "the expected per patient needs that you choose."), 
            dcc.Markdown("**2. Downloading csv file.**" + 
                         " Click on the \"Download CSV\" text below the PPE Forecast Table" + 
                         " to download a csv file of the tabulated data."
                         ),
            
            html.Br(),
            dcc.Markdown("**See the Details box below or our [publication](https://academic.oup.com/jamiaopen/advance-article/doi/10.1093/jamiaopen/ooaa045/5907917) for deeper insights.**"),
            html.Br(),
        ],
    )

def description_card4():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card4",
        children=[
            html.H5("Details on Modeling and Data", style={
            'textAlign': 'left',
            }),
            dcc.Markdown("-------"),
            dcc.Markdown("**Models**"),
            dcc.Markdown("**Exponential:** " +
                         "Initial stages of growth often appear limited only by the inherent growth rate (r) of the population or disease. " +
                         "In this way, exponential growth proceeds multiplicatively according to a simple functional form, Nt = N0ert, where " +
                         "N0 is the initial infected population size, t is the amount of passed, and Nt is the infected population size at t. " +
                         "The exponential model has been widely used to characterize the spread of COVID-19 the during initial weeks of infection." +
                         "Our application uses exponential regression to obtain " +
                         "predictions for the expected cumulative number of confirmed COVID-19 cases (N). "+
                         "This model has explained upwards of 99% of variation in the initial days or weeks of COVID-19 spread within states; " +
                         "however, it quickly begins to fail because it only allows for continued rapid growth."
                         ),
            html.Br(),
            dcc.Markdown("**Quadratic:** " +
                         "Initial stages of growth may be more rapid than that expected from the exponential model while the latter monotonic " +
                         "increase in N can proceed less rapidly than predicted by the exponential. In these cases, growth may be quadratic, " +
                         "i.e., characterized by a constant change in growth rate. Early COVID-19 studies have implicated quadratic growth in " +
                         "spread of COVID-19 and the quadratic model, to date, has continued to perform well. The quadratic function, f(x) = " +
                         "x2 + x + c, is a 2nd order polynomial that can be applied to population growth as Nt = β1t2 + β2t + N0. Our application " +
                         "uses numerical optimization of the fitted parameters, β1 and β2, to find the best fit quadratic function for a given " +
                         "time series and hence, to predict values for (N). Like the exponential model, " +
                         "the quadratic model only allows for continued growth, i.e., no saturation. Consequently, the quadratic model must " +
                         "eventually fail as COVID-19 cases saturate."
                         ),
            html.Br(),
            dcc.Markdown("**Logistic (Multiphase):** " +  
                         "Exponential growth within a population cannot continue *ad infinitum*. Instead, growth must slow as an upper limit " +
                         "is approached or as natural limitations to disease spread (e.g., immunity, contact among hosts) are encountered. " +
                         "The logistic model captures this slowing and eventual saturation, resulting in a sigmoidal or s-shaped growth curve. " +
                         "In addition to exponential and quadratic growth, early COVID-19 studies have implicated logistic growth in the spread " + 
                         "of the disease. The logistic model takes a relatively simple functional form, " +
                         "N_t=α/(1+e^(-rt) ), where α is the upper limit of N and r is the intrinsic rate of increase. Our application extends " +
                         "the logistic model to take multiple phases of logistic growth and uses numerical optimization to find the best fit parameters. "
                         ),
            html.Br(),
            dcc.Markdown("**Gaussian (Multiphase):** " +  
                         "The Gaussian (i.e., normal) distribution can provide a relatively simple and close approximation to complex epidemiological " +
                         "models. This symmetrical curve has two parameters, mean = μ, standard deviation = σ, and belongs to the family of exponential " +
                         "distributions. When used to model spread of disease, Gaussian curves are symmetrical around a climax day with the change " +
                         "in the rate of growth determining the standard deviation about the curve. Gaussian models have previously been successful " +
                         "in approximating the spread of COVID-19 in Germany. Our application extends the Gaussian model to take multiple phases of " +
                         "guassian growth and uses numerical optimization to find the best fit parameters. "
                         ),
                        
            
            html.Br(),
            html.Br(),
            dcc.Markdown("-------"),
            dcc.Markdown("**DATA**"),
            html.Br(),
            dcc.Markdown("**COVID Calculator Data.** Our COVID Calculator accesses COVID-19 data from the Johns Hopkins University Center for Systems Science and Engineering "+
                         "([JHU CSSE](https://systems.jhu.edu/)). Specifically,"+
                         "our application downloads, curates, aggregates, and stores daily reports from the JHU CSSE public "+
                         "[GitHub repository](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports). These daily reports contain cumulative numbers of confirmed cases, and cumulative numbers of reported "+
                         "deaths and recoveries for counties, states, provinces, and nations reported since January 22nd, 2020. For select models, our "+
                         "application uses population sizes for US states and territories based on data from the US Census Bureau (2010 – 2019). "+
                         "Our application also uses dates of COVID-19 arrival in US states and territories based on data available from state and territory "+
                         "governmental agencies (e.g., Departments of Health)."
                         ),
            html.Br(),
            dcc.Markdown("**Hospitalization and Testing Data.** Tabs on Trends in Testing and Trends in Hospitalization use a combination of the "+
                         "JHU CSSE daily reports data as well as data from The Atlantic's [COVID Tracking Project](https://covidtracking.com/). "+
                         "Description of the COVID tracking project's data can be found [here](https://covidtracking.com/data/download)."
                         ),

        ],
    )





def generate_control_card1():
    
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card1",
        children=[
            html.P("Select a state or territory"),
            dcc.Dropdown(
                id="location-select1",
                options=[{"label": i, "value": i} for i in locations],
                value='Illinois',
                style={
                    'width': '250px', 
                    'font-size': "100%",
                    }
            ),
            
            html.Br(),
            html.P("Select a county or other area"),
            dcc.Dropdown(
                id="county-select1",
                options=[{"label": i, "value": i} for i in counties],
                value='Entire state or territory',
                style={
                    'width': '250px', 
                    'font-size': "100%",
                    }
            ),
            
            html.Br(),
            html.P("Select a model"),
            dcc.Dropdown(
                id="model-select1",
                options=[{"label": i, "value": i} for i in models],
                value=models[0],
            ),
            html.P("Most of these models are intensive to compute. So, allow the current model to finish before picking another model, county, or state. Otherwise, the app could time-out and you will need to refresh the page."),
            
            #html.Br(),
            #html.Div(
            #    id="add-forecast",
            #    children=html.Button(id="add-forecast1", children="Plot prior day's forecast", n_clicks=10),
            #),
            #html.P("Each click will add a previous day's forecast to the top figure."),
            
            html.Br(),
            html.Hr(),
            
            html.Br(),
            html.H5("Select Hospital Variables"),
            
            
            html.Br(),
            html.Div(id='ICU beds1-container'),
            dcc.Slider(
                id="ICU beds1",
                min=0,
                max=500,
                value=300,
                step = 1),  
            
            html.Br(),
            html.Div(id='nonICU beds1-container'),
            dcc.Slider(
                id="nonICU beds1",
                min=0,
                max=500,
                value=300,
                step=1),
            
            html.Br(),
            html.Div(id='visits1-container'),
            dcc.Slider(
                id="visits1",
                min=0,
                max=100,
                value=10),    
            
            html.Br(),
            html.Div(id='admits1-container'),
            dcc.Slider(
                id="admits1",
                min=0,
                max=100,
                value=10),
            
            html.Br(),
            html.Div(id='percent ICU1-container'),
            dcc.Slider(
                id="percent ICU1",
                min=0,
                max=100,
                value=10),
            
            html.Br(),
            html.Div(id='transfers1-container'),
            dcc.Slider(
                id="transfers1",
                min=0,
                max=100,
                value=10), 
            
            html.Br(),
            html.Div(id='percent transferICU1-container'),
            dcc.Slider(
                id="percent transferICU1",
                min=0,
                max=100,
                value=10),
            
            html.Br(),
            html.Div(id='on vent1-container'),
            dcc.Slider(
                id="on vent1",
                min=0,
                max=100,
                value=10),  
            
            html.Br(),
            html.Div(id='non-ICU LOS1-container'),
            dcc.Slider(
                id="non-ICU LOS1",
                min=1,
                max=14,
                value=4),
            
            html.Br(),
            html.Div(id='ICU LOS1-container'),
            dcc.Slider(
                id="ICU LOS1",
                min=1,
                max=14,
                value=4),
            
            html.Br(),
            html.Div(id='mortality1-container'),
            dcc.Slider(
                id="mortality1",
                min=0,
                max=100,
                value=10),
            
            html.Br(),
            html.Div(id='time lag1-container'),
            dcc.Slider(
                id="time lag1",
                min=1,
                max=14,
                value=4),
            
            
            html.Br(),
            html.Hr(),
            
            html.Br(),
            html.Br(),
            html.H5("Select Daily PPE Needs (per patient)"),
            
            html.Br(),
            html.Div(id='GLOVE SURGICAL-container'),
            dcc.Slider(
                id="gloves1",
                min=0,
                max=100,
                value=2),
            
            
            html.Br(),
            html.Div(id='GLOVE EXAM NITRILE-container'),
            dcc.Slider(
                id="gloves2",
                min=0,
                max=1000,
                value=260),
            
            
            html.Br(),
            html.Div(id='GLOVE EXAM VINYL-container'),
            dcc.Slider(
                id="gloves3",
                min=0,
                max=100,
                value=10),
            
            
            html.Br(),
            html.Div(id='MASK FACE PROC ANTI FOG-container'),
            dcc.Slider(
                id="mask1",
                min=0,
                max=100,
                value=45),
            
            
            html.Br(),
            html.Div(id='MASK PROC FLUID RESISTANT-container'),
            dcc.Slider(
                id="mask2",
                min=0,
                max=100,
                value=1),
            
            
            html.Br(),
            html.Div(id='GOWN ISOLATION XL YELLOW-container'),
            dcc.Slider(
                id="gown1",
                min=0,
                max=20,
                value=2),
            
            
            html.Br(),
            html.Div(id='MASK SURG ANTI FOG W/FILM-container'),
            dcc.Slider(
                id="mask3",
                min=0,
                max=20,
                value=1),
            
            
            html.Br(),
            html.Div(id='SHIELD FACE FULL ANTI FOG-container'),
            dcc.Slider(
                id="shield1",
                min=0,
                max=20,
                value=1),
            
            
            html.Br(),
            html.Div(id='RESP PART FILTER REG-container'),
            dcc.Slider(
                id="resp1",
                min=0,
                max=100,
                value=11),
            
            
            html.Br(),
            html.Br(),
            html.Div(
                id="reset-btn-outer1",
                children=html.Button(id="reset-btn1", children="Reset Plots1", n_clicks=0),
            ),
        ],
    )




def generate_control_card2():
    
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card2",
        children=[
            html.P("Select a state or territory"),
            dcc.Dropdown(
                id="location-select2",
                options=[{"label": i, "value": i} for i in locations],
                value='Illinois',
                style={
                    'width': '250px', 
                    'font-size': "100%",
                    }
            ),
            
            html.Br(),
            html.P("Select a county or other area"),
            dcc.Dropdown(
                id="county-select2",
                options=[{"label": i, "value": i} for i in counties],
                value='Entire state or territory',
                style={
                    'width': '250px', 
                    'font-size': "100%",
                    }
            ),
            
            html.Br(),
            html.P("Select a model"),
            dcc.Dropdown(
                id="model-select2",
                options=[{"label": i, "value": i} for i in models],
                value=models[0],
            ),
            
            
            html.Br(),
            html.Hr(),
            
            html.H5("Employee variables"),
            
            html.P("No. of employees"),
            dcc.Input(
                id="employees",
                placeholder=1000,
                value=1000,
                type='number',
            ),
            
            html.Br(),
            html.Br(),
            html.P("No. of employees needed per covid patient"),
            dcc.Input(
                id="employees per patient or per bed",
                placeholder=3,
                value=3,
                type='number',
            ),
            
            html.Br(),
            html.Br(),
            html.P("Minimum furlough days after an employee tests positive"),
            dcc.Input(
                id="furlough",
                placeholder=10,
                value=10,
                type='number',
            ),
            
            html.Br(),
            html.Br(),
            html.P('Relative positivity rate'),
            dcc.Slider(
                id="inc_rate",
                min=0,
                max=200,
                value=100,
                step=1,
                marks={
                    0: '0%',
                    50: '50%',
                    100: '100%',
                    150: '150%',
                    200: '200%',
                },
                ),
            html.Div(id='incidence rate-container'),
            
            html.Br(),
            html.Br(),
            html.Div(
                id="reset-btn-outer2",
                children=html.Button(id="reset-btn2", children="Reset Plots2", n_clicks=0),
            ),
        ],
    )



def generate_model_forecasts(loc, county, model, reset):
    
        
    new_cases = []
    ForecastDays = 60
    
    col_names =  ['obs_y', 'pred_y', 'forecasted_y', 'pred_dates', 'forecast_dates', 
                  'label', 'obs_pred_r2', 'model', 'focal_loc', 
                  'pred_clr', 'fore_clr']
        
    fits_df = pd.DataFrame(columns = col_names)

    if county == 'Entire state or territory':
        
        url = 'https://raw.githubusercontent.com/klocey/StateCovidData/main/data/COVID-CASES-DF.txt'
        locs_df = pd.read_csv(url, sep='\t')

        locs_df = locs_df[locs_df['Country/Region'] == 'US']
        locs_df = locs_df[~locs_df['Province/State'].isin(['US', 'American Samoa', 'Northern Mariana Islands',
                            'Diamond Princess', 'Grand Princess', 'Recovered', 'United States Virgin Islands', 
                            'Virgin Islands, U.S.', 'Wuhan Evacuee'])]
        
        locs_df.drop(columns=['Unnamed: 0'], inplace=True)


        df_sub = locs_df[locs_df['Province/State'] == loc]
        locs_df = 0
        ArrivalDate = statepops[statepops['Province/State'] == loc]['Date_of_first_reported_infection'].tolist()
        ArrivalDate = ArrivalDate[0]

    
    else:
        
        try:
            
            url = 'https://raw.githubusercontent.com/klocey/StateCovidData/main/data/' + loc + '-' + county + '-' + 'COVID-CASES.txt'
            
            df_sub = pd.read_csv(url, sep='\t')
            
            #counties_df.drop(['Unnamed: 0'], axis=1, inplace=True)
            df_sub = df_sub.filter(items=['date', 'Confirmed'], axis=1)
            df_sub = df_sub.set_index('date').transpose()
            df_sub = df_sub.reset_index(drop=True)
            #df_sub.drop(['date'], axis=1, inplace=True)
            
            df_sub['Province/State'] = loc
            df_sub['Country/Region'] = 'US'
            df_sub['Lat'] = 0
            df_sub['Long'] = 0
            
            col = df_sub.pop('Long')
            df_sub.insert(0, col.name, col)
            
            col = df_sub.pop('Lat')
            df_sub.insert(0, col.name, col)
            
            col = df_sub.pop('Country/Region')
            df_sub.insert(0, col.name, col)
            
            col = df_sub.pop('Province/State')
            df_sub.insert(0, col.name, col)

            
        except:
            PopSize = statepops[statepops['Province/State'] == loc]['PopSize'].tolist()
            PopSize = PopSize[0]
            
            url = 'https://raw.githubusercontent.com/klocey/StateCovidData/main/data/COVID-CASES-DF.txt'
            locs_df = pd.read_csv(url, sep='\t')

            locs_df = locs_df[locs_df['Country/Region'] == 'US']
            locs_df = locs_df[~locs_df['Province/State'].isin(['US', 'American Samoa', 'Northern Mariana Islands',
                                                            'Diamond Princess', 'Grand Princess', 'Recovered', 
                                                             'United States Virgin Islands', 'Virgin Islands, U.S.',
                                                            'Wuhan Evacuee'])]
            
            locs_df.drop(columns=['Unnamed: 0'], inplace=True)

            df_sub = locs_df[locs_df['Province/State'] == loc]
            locs_df = 0
            
        
    # add 1 to number of forecast days for indexing purposes
    ForecastDays = int(ForecastDays+1)
                
    # get column labels, will filter below to extract dates
    yi = list(df_sub)
        
    obs_y_trunc = []
    fore_clrs =  ['purple',  'mediumorchid', 'plum', 'blue', 'deepskyblue', 
                  'darkturquoise', 'green', 'limegreen', 'gold', 'orange', 'red']
    pred_clrs = ['0.0', '0.1', '0.2', '0.25', '0.3', '0.35', '0.4', '0.5',
                 '0.6', '0.7', '0.8']
    
    conditions = []        
    for i, j in enumerate(list(range(-10, 1))):
            
        if j == 0:
            # get dates for today's predictions/forecast
            obs_y_trunc = df_sub.iloc[0,4:].values
        else:
            # get dates for previous days predictions/forecast
            obs_y_trunc = df_sub.iloc[0,4:j].values
            
        
        ii = 0
        while obs_y_trunc[ii] == 0: ii+=1
        y = obs_y_trunc[ii:]
        
        y = list(y)
        if y != sorted(y):
            for ii, val in enumerate(y):
                if ii == 0: 
                    continue
                elif val < y[ii-1]:
                    y[ii] = y[ii-1]
                    
        
        # declare x as a list of integers from 0 to len(y)
        x = list(range(len(y)))
        iterations = 2
        # Call function to use chosen model to obtain:
        #    r-square for observed vs. predicted
        #    predicted y-values
        #    forecasted x and y values
        
        condition = [x, y, model, ForecastDays, j, iterations]
        conditions.append(condition)
    
    pool = Pool()
    results = pool.map(fxns.fit_curve, conditions)
    pool.close() 
    pool.join()
    
    
    for i, j in enumerate(list(range(-10, 1))):
        pred_clr = pred_clrs[i]
        fore_clr = fore_clrs[i]
        
        # designature plot label for legend
        if j == 0:
            label='Current forecast'
            
        else:
            label = str(-j)+' day old forecast'
        
        if j == 0:
            # get dates for today's predictions/forecast
            DATES = yi[4:]
            obs_y_trunc = df_sub.iloc[0,4:].values
        else:
            # get dates for previous days predictions/forecast
            DATES = yi[4:j]
            obs_y_trunc = df_sub.iloc[0,4:j].values
            
        
        ii = 0
        while obs_y_trunc[ii] == 0: ii+=1
        y = obs_y_trunc[ii:]
        dates = DATES[ii:]
        
            
        result = results[i]
        
        obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y, params = result
        
        # convert y values to numpy array
        y = np.array(y)

        # because it isn't based on a best fit line,
        # and the y-intercept is forced through [0,0]
        # a model can perform so poorly that the 
        # observed vs predicted r-square is negative (a nonsensical value)
        # if this happens, report the r-square as 0.0
        if obs_pred_r2 < 0:
            obs_pred_r2 = 0.0

        # convert any y-values (observed, predicted, or forecasted)
        # that are less than 0 (nonsensical values) to 0.
        y[y < 0] = 0
        pred_y = np.array(pred_y)
        pred_y[pred_y < 0] = 0

        forecasted_y = np.array(forecasted_y)
        forecasted_y[forecasted_y < 0] = 0
            
        # number of from ArrivalDate to end of forecast window
        #numdays = len(forecasted_x)
        latest_date = pd.to_datetime(dates[-1])
        first_date = pd.to_datetime(dates[0])

        # get the date of the last day in the forecast window
        future_date = latest_date + datetime.timedelta(days = ForecastDays-1)
            
        # get all dates from ArrivalDate to the last day in the forecast window
        fdates = pd.date_range(start=first_date, end=future_date)
        fdates = fdates.strftime('%m/%d/%Y')
            
        # designature plot label for legend
        if j == 0:
            label='Current forecast'
            
        else:
            label = str(-j)+' day old forecast'
            
            
        if label == 'Current forecast':
            for ii, val in enumerate(forecasted_y):
                if ii > 0:
                    if forecasted_y[ii] - forecasted_y[ii-1] > 0:
                        new_cases.append(forecasted_y[ii] - forecasted_y[ii-1])
                    else:
                        new_cases.append(0)
                if ii == 0:
                    new_cases.append(forecasted_y[ii])
                        
                
        # get dates from ArrivalDate to the current day
        dates = pd.date_range(start=first_date, end=latest_date)
        dates = dates.strftime('%m/%d/%Y')
            
            
        output_list = [y.tolist(), pred_y.tolist(), forecasted_y.tolist(), dates, fdates,
                       label, obs_pred_r2, model, loc, #PopSize, ArrivalDate, 
                       pred_clr, fore_clr]
            
        fits_df.loc[len(fits_df)] = output_list

    dates = 0
    df_sub = 0
    output_list = 0
    fdates = 0
    pred_y = 0
    forecasted_y = 0
    output_list = 0
    
    fits_df = fits_df.to_json()
    return fits_df



    

def generate_model_forecast_plot(fits_df, reset):
    fits_df = pd.read_json(fits_df)
    
    ForecastDays = 60
    #labels = fits_df['label'].tolist()
    
    labels = ['Current forecast', '1 day old forecast', 
              '2 day old forecast', '3 day old forecast', '4 day old forecast', 
              '5 day old forecast', '6 day old forecast', '7 day old forecast', 
              '8 day old forecast', '9 day old forecast', '10 day old forecast']
    
    fig_data = []
    
    obs_y = []
    fdates = []
    
    for i, label in enumerate(labels):
            
        try:
            sub_df = fits_df[fits_df['label'] == label]
            r2 = sub_df['obs_pred_r2'].iloc[0]
            
            dates = sub_df['pred_dates'].iloc[0]
            clr = sub_df['pred_clr'].iloc[0]
            obs_y = sub_df['obs_y'].iloc[0]
            if label == 'Current forecast':
                fig_data.append(
                    go.Scatter(
                        x=dates,
                        y=obs_y,
                        mode="markers",
                        name='Observed',
                        opacity=0.75,
                        marker=dict(color='#262626', size=10)
                    )
                )
            
            
            fdates = sub_df['forecast_dates'].iloc[0]
            forecasted_y = sub_df['forecasted_y'].iloc[0]
            clr = sub_df['fore_clr'].iloc[0]
            #focal_loc = sub_df['focal_loc'].iloc[0]
            #popsize = sub_df['PopSize'].iloc[0]
                
            pred_y = sub_df['pred_y'].iloc[0]
            # plot forecasted y values vs. dates
            l = int(len(pred_y)+ForecastDays)
                
            forecasted_y = forecasted_y[0 : l]
            fdates = fdates[0 : l]
            
            fig_data.append(
                go.Scatter(
                    x=fdates,
                    y=forecasted_y,
                    name=label + ': r<sup>2</sup> = ' + str(np.round(r2, 3)),
                    mode="lines",
                    line=dict(color=clr, width=2)
                )
            )
        except:
            pass
        

    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(
                title=dict(
                    text="<b>Date</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            yaxis=dict(
                title=dict(
                    text="<b>Cumulative COVID-19 cases</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                        
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            margin=dict(l=60, r=30, b=10, t=40),
            showlegend=True,
            height=400,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),
    )
    
    figure.update_yaxes(range=[0, 1.5*max(obs_y)])
    figure.update_layout(
        title=dict(text="r<sup>2</sup> values pertain to the fits of models (colored lines) to the previous 30 days of observed data.",
                   font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                    ),
                    ),
        )
    
    dates = 0
    sub_df = 0
    fits_df = 0
    fdates = 0
    pred_y = 0
    forecasted_y = 0
    
    return figure



def generate_model_forecast_table(fits_df, reset):
    df = pd.read_json(fits_df)
    df_table = pd.DataFrame()
    
    col0 = df.forecast_dates.iloc[-1]
    max_len = len(col0)
    
    col1 = df.obs_y.iloc[-1]    
    zs = [0] * (max_len - len(col1))
    col1.extend(zs)
    
    col2 = df.forecasted_y.iloc[-1]
    zs = [0] * (max_len - len(col2))
    col2.extend(zs)
    
    col3 = df.forecasted_y.iloc[-2]
    zs = [0] * (max_len - len(col3))
    col3.extend(zs)
    
    col4 = df.forecasted_y.iloc[-3]
    zs = [0] * (max_len - len(col4))
    col4.extend(zs)
    
    col5 = df.forecasted_y.iloc[-4]
    zs = [0] * (max_len - len(col5))
    col5.extend(zs)
    
    col6 = df.forecasted_y.iloc[-5]
    zs = [0] * (max_len - len(col6))
    col6.extend(zs)
    
    col7 = df.forecasted_y.iloc[-6]
    zs = [0] * (max_len - len(col7))
    col7.extend(zs)
    
    col8 = df.forecasted_y.iloc[-7]
    zs = [0] * (max_len - len(col8))
    col8.extend(zs)
    
    col9 = df.forecasted_y.iloc[-8]
    zs = [0] * (max_len - len(col9))
    col9.extend(zs)
    
    col10 = df.forecasted_y.iloc[-9]
    zs = [0] * (max_len - len(col10))
    col10.extend(zs)
    
    col11 = df.forecasted_y.iloc[-10]
    zs = [0] * (max_len - len(col11))
    col11.extend(zs)
    
    col12 = df.forecasted_y.iloc[-11]
    zs = [0] * (max_len - len(col12))
    col12.extend(zs)

    
    df_table['Dates'] = col0
    df_table['Confirmed cases'] = col1
    df_table['Current forecast'] = np.round(col2, 0)
    df_table['1 day ago'] = np.round(col3, 0)
    df_table['2 days ago'] = np.round(col4, 0)
    df_table['3 days ago'] = np.round(col5, 0)
    df_table['4 days ago'] = np.round(col6, 0)
    df_table['5 days ago'] = np.round(col7, 0)
    df_table['6 days ago'] = np.round(col8, 0)
    df_table['7 days ago'] = np.round(col9, 0)
    df_table['8 days ago'] = np.round(col10, 0)
    df_table['9 days ago'] = np.round(col11, 0)
    df_table['10 days ago'] = np.round(col12, 0)
    
    csv_string = df_table.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    
    col0, col1, col1, col3, col4, col5, col6, col7, col8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    col9, col10, col11, col12 = 0, 0, 0, 0
    df_table = 0
    fits_df = 0
    
    return csv_string
    
    
    


        
def generate_patient_census(loc, county, model, icu_beds, nonicu_beds, per_loc, per_admit, 
    per_cc, LOS_cc, LOS_nc, per_vent, TimeLag, transfers, per_ICU_transfer, 
    mortality, GLOVE_SURGICAL, GLOVE_EXAM_NITRILE, GLOVE_EXAM_VINYL, 
    MASK_FACE_PROC_ANTI_FOG, MASK_PROC_FLUID_RESISTANT, GOWN_ISOLATION_XL_YELLOW, 
    MASK_SURG_ANTI_FOG_FILM, SHIELD_FACE_FULL_ANTI_FOG, RESP_PART_FILTER_REG,
    reset):
    
    new_cases = []
    ForecastDays = 60

    
    PopSize = statepops[statepops['Province/State'] == loc]['PopSize'].tolist()
    PopSize = PopSize[0]
    

    if county == 'Entire state or territory':
        url = 'https://raw.githubusercontent.com/klocey/StateCovidData/main/data/COVID-CASES-DF.txt'
        locs_df = pd.read_csv(url, sep='\t')

        locs_df = locs_df[locs_df['Country/Region'] == 'US']
        locs_df = locs_df[~locs_df['Province/State'].isin(['US', 'American Samoa', 'Northern Mariana Islands',
                                                        'Diamond Princess', 'Grand Princess', 'Recovered', 
                                                         'United States Virgin Islands', 'Virgin Islands, U.S.',
                                                        'Wuhan Evacuee'])]
        
        locs_df.drop(columns=['Unnamed: 0'], inplace=True)

        df_sub = locs_df[locs_df['Province/State'] == loc]
        ArrivalDate = statepops[statepops['Province/State'] == loc]['Date_of_first_reported_infection'].tolist()
        ArrivalDate = ArrivalDate[0]
        locs_df = 0
        
    
    else:
        
        try:
            url = 'https://raw.githubusercontent.com/klocey/StateCovidData/main/data/' + loc + '-' + county + '-' + 'COVID-CASES.txt'
            df_sub = pd.read_csv(url, sep='\t') #index_col=0)

            #counties_df.drop(['Unnamed: 0'], axis=1, inplace=True)
            df_sub = df_sub.filter(items=['date', 'Confirmed'], axis=1)
            df_sub = df_sub.set_index('date').transpose()
            df_sub = df_sub.reset_index(drop=True)
            #df_sub.drop(['date'], axis=1, inplace=True)
            
            df_sub['Province/State'] = loc
            df_sub['Country/Region'] = 'US'
            df_sub['Lat'] = 0
            df_sub['Long'] = 0
            
            col = df_sub.pop('Long')
            df_sub.insert(0, col.name, col)
            
            col = df_sub.pop('Lat')
            df_sub.insert(0, col.name, col)
            
            col = df_sub.pop('Country/Region')
            df_sub.insert(0, col.name, col)
            
            col = df_sub.pop('Province/State')
            df_sub.insert(0, col.name, col)
            
    
            ArrivalDate = np.nan
            
            
        except:
            url = 'https://raw.githubusercontent.com/klocey/StateCovidData/main/data/COVID-CASES-DF.txt'
            locs_df = pd.read_csv(url, sep='\t')

            locs_df = locs_df[locs_df['Country/Region'] == 'US']
            locs_df = locs_df[~locs_df['Province/State'].isin(['US', 'American Samoa', 'Northern Mariana Islands',
                                                            'Diamond Princess', 'Grand Princess', 'Recovered', 
                                                             'United States Virgin Islands', 'Virgin Islands, U.S.',
                                                            'Wuhan Evacuee'])]
            
            locs_df.drop(columns=['Unnamed: 0'], inplace=True)

            df_sub = locs_df[locs_df['Province/State'] == loc]
            ArrivalDate = statepops[statepops['Province/State'] == loc]['Date_of_first_reported_infection'].tolist()
            ArrivalDate = ArrivalDate[0]
            locs_df = 0
            
      
    # add 1 to number of forecast days for indexing purposes
    ForecastDays = int(ForecastDays+1)
        
    # get column labels, will filter below to extract dates
    yi = list(df_sub)
        
    obs_y_trunc = []
    DATES = yi[4:]
    obs_y_trunc = df_sub.iloc[0,4:].values
    
    ii = 0
    while obs_y_trunc[ii] == 0: ii+=1
    y = obs_y_trunc[ii:]
    dates = DATES[ii:]
        
    y = list(y)
    if y != sorted(y):
        for ii, val in enumerate(y):
            if ii == 0: 
                continue
            elif val < y[ii-1]:
                y[ii] = y[ii-1]
                    
    # declare x as a list of integers from 0 to len(y)
    x = list(range(len(y)))

    # Call function to use chosen model to obtain:
    #    r-square for observed vs. predicted
    #    predicted y-values
    #    forecasted x and y values
    iterations = 2
    
    condition = [x, y, model, ForecastDays, 0, iterations]
    result = fxns.fit_curve(condition)
    obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y, params = result
    
    # convert y values to numpy array
    y = np.array(y)

    # because it isn't based on a best fit line,
    # and the y-intercept is forced through [0,0]
    # a model can perform so poorly that the 
    # observed vs predicted r-square is negative (a nonsensical value)
    # if this happens, report the r-square as 0.0
    if obs_pred_r2 < 0:
        obs_pred_r2 = 0.0

    # convert any y-values (observed, predicted, or forecasted)
    # that are less than 0 (nonsensical values) to 0.
    y[y < 0] = 0
    forecasted_y = np.array(forecasted_y)
    forecasted_y[forecasted_y < 0] = 0
            
    # number of from ArrivalDate to end of forecast window
    #numdays = len(forecasted_x)
    latest_date = pd.to_datetime(dates[-1])
    first_date = pd.to_datetime(dates[0])

    # get the date of the last day in the forecast window
    future_date = latest_date + datetime.timedelta(days = ForecastDays-1)
    
    # get all dates from ArrivalDate to the last day in the forecast window
    fdates = pd.date_range(start=first_date, end=future_date)
    fdates = fdates.strftime('%m/%d/%Y')
    
    # designature plot label for legend
    for i, val in enumerate(forecasted_y):
        if i > 0:
            if forecasted_y[i] - forecasted_y[i-1] > 0:
                new_cases.append(forecasted_y[i] - forecasted_y[i-1])
            else:
                new_cases.append(0)
        if i == 0:
            new_cases.append(0)
            
    new_obs = []
    for i, val in enumerate(y):
        if i > 0:
            if y[i] - y[i-1] > 0:
                new_obs.append(y[i] - y[i-1])
            else:
                new_obs.append(0)
        if i == 0:
            new_obs.append(0)

                
    # get dates from ArrivalDate to the current day
    dates = pd.date_range(start=first_date, end=latest_date)
    dates = dates.strftime('%m/%d/%Y')
            
    # declare column labels
    col_labels = ['date', 'Total cases', 'New visits', 
                  'New admits', 'All COVID', 'Non-ICU', 'ICU', 'Vent',
                  'Discharged from ICU deceased', 'Discharged from ICU alive',
                  'Discharged from non-ICU alive']
    
    # row labels are the dates
    row_labels = fdates.tolist()
        
    #### Inclusion of time lag
    # time lag is modeled as a Poisson distributed 
    # random variable with a mean chosen by the user (TimeLag)
    new_cases_lag = []
    x = list(range(len(forecasted_y)))
    for i in new_cases:
        lag_pop = i*poisson.pmf(x, TimeLag)
        new_cases_lag.append(lag_pop)
         
    # Declare a list to hold time-staggered lists
    # This will allow the time-lag effects to
    # be summed across rows (days)
    lol = []
    for i, daily_vals in enumerate(new_cases_lag):
        # number of indices to pad in front
        fi = [0]*i
        diff = len(new_cases) - len(fi)
        # number of indices to pad in back
        bi = [0]*diff
        ls = list(fi) + list(daily_vals) + list(bi)
        lol.append(np.array(ls))
        
    # convert the list of time-staggered lists to an array
    ar = np.array(lol)
        
    # get the time-lagged sum of visits across days
    ts_lag = np.sum(ar, axis=0)
    # upper truncate for the number of days in observed y values
    ts_lag = ts_lag[:len(new_cases)]
        
    # row labels are the dates
    row_labels = fdates.tolist()  

    #### Construct arrays for critical care and non-critical care patients

    # Use lognormal cdf to parameterize patient turnover
    sigma = 0.5
    n_cc = np.log(LOS_cc) - (sigma**2)/2
    n_nc = np.log(LOS_nc) - (sigma**2)/2
    
    x_vars = np.array(list(range(1, len(fdates)+1)))
    
    p_nc = 0.5 + 0.5 * sc.special.erf((np.log(x_vars) - n_nc)/(2**0.5*sigma))
    p_cc = 0.5 + 0.5 * sc.special.erf((np.log(x_vars) - n_cc)/(2**0.5*sigma))
    
    # Initiate lists to hold numbers of critical care and non-critical care patients
    # who are expected as new admits (index 0), as 1 day patients, 2 day patients, etc.
    LOScc = np.zeros(len(fdates))
    LOScc[0] = ts_lag[0] * (0.01 * per_cc) * (0.01 * per_admit) * (0.01 * per_loc)
    LOSnc = np.zeros(len(fdates))
    LOSnc[0] =  ts_lag[0] * (1-(0.01 * per_cc)) * (0.01 * per_admit) * (0.01 * per_loc)
        
    total_nc = []
    total_cc = []
    discharged_dead_cc = []
    discharged_alive_cc = []
    discharged_alive_nc = []
    
    # Roll up patient carry-over into lists of total critical care and total
    # non-critical patients expected
    
    for i, day in enumerate(fdates):
        
        tr = transfers/(1+np.exp(-0.1*transfers+4))
        
        discharged_cc = LOScc * p_cc
        discharged_nc = LOSnc * p_nc
        
        d1 = np.sum(0.01 * mortality * discharged_cc)
        d2 = np.sum((1 - 0.01 * mortality) * discharged_cc)
        d3 = np.sum(discharged_nc)
        #a1 = np.sum(inactive)
        
        discharged_dead_cc.append(d1)
        discharged_alive_cc.append(d2)
        discharged_alive_nc.append(d3)
        
        LOScc = LOScc - discharged_cc
        LOSnc = LOSnc - discharged_nc
            
        LOScc = np.roll(LOScc, shift=1)
        LOSnc = np.roll(LOSnc, shift=1)
            
        LOScc[0] = ts_lag[i] * (0.01 * per_cc) * (0.01 * per_admit) * (0.01 * per_loc) + ((0.01 * per_ICU_transfer) * tr)
        LOSnc[0] = ts_lag[i] * (1 - (0.01 * per_cc)) * (0.01 * per_admit) * (0.01 * per_loc) + ((1 - 0.01 * per_ICU_transfer) * tr)
        
        total_nc.append(np.sum(LOSnc))
        total_cc.append(np.sum(LOScc))
        
    
    cells = []
    
    for i in range(len(row_labels)):
            
        val = ts_lag[i]
        cell = [row_labels[i],
                int(np.round(forecasted_y[i])), 
                int(np.round(val * (per_loc * 0.01))),
                int(np.round((0.01 * per_admit) * val * (per_loc * 0.01))),
                int(np.round(total_nc[i] + total_cc[i])), 
                int(np.round(total_nc[i])),
                int(np.round(total_cc[i])), 
                int(np.round(total_cc[i]*(0.01*per_vent))),
                int(discharged_dead_cc[i]),
                int(discharged_alive_cc[i]),
                int(discharged_alive_nc[i]),
                ]
        
        cells.append(cell)
        
    # Add the row to the dataframe
    census_df = pd.DataFrame.from_records(cells, columns=col_labels)    
    
    #### Construct arrays for critical care and non-critical care patients
        
    # All covid patients expected in house on each forecasted day. PUI is just a name here
    
    PUI_COVID = np.array(total_nc) + np.array(total_cc) 
    # Preparing to add new visits, fraction of new cases visiting your hospital = 0.01 * per_loc 
    new_visits_your_hospital = ts_lag * (0.01 * per_loc)
    # Add number of new visits to number of in house patients
    PUI_COVID = PUI_COVID + new_visits_your_hospital
        
    glove_surgical = np.round(GLOVE_SURGICAL * PUI_COVID).astype('int')
    glove_nitrile = np.round(GLOVE_EXAM_NITRILE * PUI_COVID).astype('int')
    glove_vinyl = np.round(GLOVE_EXAM_VINYL * PUI_COVID).astype('int')
    face_mask = np.round(MASK_FACE_PROC_ANTI_FOG * PUI_COVID).astype('int')
    procedure_mask = np.round(MASK_PROC_FLUID_RESISTANT * PUI_COVID).astype('int')
    isolation_gown = np.round(GOWN_ISOLATION_XL_YELLOW * PUI_COVID).astype('int')
    surgical_mask = np.round(MASK_SURG_ANTI_FOG_FILM * PUI_COVID).astype('int')
    face_shield = np.round(SHIELD_FACE_FULL_ANTI_FOG * PUI_COVID).astype('int')
    respirator = np.round(RESP_PART_FILTER_REG * PUI_COVID).astype('int')
        
    
    ppe_ls =[[glove_surgical, 'GLOVE SURGICAL', 'red'],
             [glove_nitrile, 'GLOVE EXAM NITRILE', 'orange'],
             [glove_vinyl, 'GLOVE EXAM VINYL', 'goldenrod'],
             [face_mask, 'MASK FACE PROCEDURE ANTI FOG', 'limegreen'],
             [procedure_mask, 'MASK PROCEDURE FLUID RESISTANT', 'green'],
             [isolation_gown, 'GOWN ISOLATION XLARGE YELLOW', 'cornflowerblue'],
             [surgical_mask, 'MASK SURGICAL ANTI FOG W/FILM', 'blue'],
             [face_shield, 'SHIELD FACE FULL ANTI FOG', 'plum'],
             [respirator, 'RESPIRATOR PARTICULATE FILTER REG', 'darkviolet']]
        
    col_labels = [ppe_ls[0][1], ppe_ls[1][1], ppe_ls[2][1], 
                  ppe_ls[3][1], ppe_ls[4][1], ppe_ls[5][1],
                  ppe_ls[6][1], ppe_ls[7][1], ppe_ls[8][1]]

    
    cells = []
    for i in range(len(row_labels)):
        
        cell = [ppe_ls[0][0][i], ppe_ls[1][0][i], ppe_ls[2][0][i], 
                ppe_ls[3][0][i], ppe_ls[4][0][i], ppe_ls[5][0][i],
                ppe_ls[6][0][i], ppe_ls[7][0][i], ppe_ls[8][0][i]]
            
        cells.append(cell)
        
    # Add the row to the dataframe
    ppe_df = pd.DataFrame.from_records(cells, columns=col_labels)     
    
    for col in list(ppe_df):
        census_df[col] = ppe_df[col].tolist()
    
    
    census_df = census_df.to_json()
    
    dates = 0
    df_sub = 0
    fdates = 0
    forecasted_y = 0
    ppe_ls = 0
    col_labels = 0
    cells = 0
    ppe_df = 0
    glove_surgical = 0
    glove_nitrile = 0
    glove_vinyl = 0
    face_mask = 0
    procedure_mask = 0
    isolation_gown = 0
    surgical_mask = 0
    face_shield = 0
    respirator = 0
    PUI_COVID = 0
    new_visits_your_hospital = 0
    PUI_COVID = 0
    cell = 0
    total_nc = 0
    total_cc = 0
    discharged_dead_cc = 0
    discharged_alive_cc = 0
    discharged_alive_nc = 0
    ar = 0
    ts_lag = 0
    row_labels = 0
    n_cc = 0
    n_nc = 0
    x_vars = 0
    p_nc = 0
    p_cc = 0    
    LOScc = 0
    LOSnc = 0
    lag_pop = 0
    new_cases_lag = 0
    lol = 0
    
    return census_df





def generate_new_and_active_cases(df, loc, county, model, reset):
    
    df = pd.read_json(df)
    df = df[df['label'] == 'Current forecast']
    
    #df['forecast_dates'] = df['forecast_dates'].dt.strftime('%m/%d')#.values.tolist()
    fdates = df['forecast_dates'].iloc[0]
    y = df['obs_y'].iloc[0]
    forecasted_y = df['forecasted_y'].iloc[0]
    
    new_cases = []
    ForecastDays = 60
    
    # add 1 to number of forecast days for indexing purposes
    ForecastDays = int(ForecastDays+1)
    
    # designature plot label for legend
    for i, val in enumerate(forecasted_y):
        if i > 0:
            if forecasted_y[i] - forecasted_y[i-1] > 0:
                new_cases.append(forecasted_y[i] - forecasted_y[i-1])
            else:
                new_cases.append(0)
        if i == 0:
            new_cases.append(0)
            
    new_obs = []
    for i, val in enumerate(y):
        if i > 0:
            if y[i] - y[i-1] > 0:
                new_obs.append(y[i] - y[i-1])
            else:
                new_obs.append(0)
        if i == 0:
            new_obs.append(0)

                
    # declare column labels
    col_labels = ['date', 'Active cases', 'New cases (Forecasted)', 'New cases (Observed)']
    
    # row labels are the dates
    row_labels = fdates

    #### Construct arrays for critical care and non-critical care patients
    total_active = []
    
    ## Use lognormal to parameterize turnover of active cases
    x_vars = np.array(list(range(1, len(fdates)+1)))
    
    sigma = 0.1
    n_active = np.log(14) - (sigma**2)/2
    p_active = 0.5 + 0.5 * sc.special.erf((np.log(x_vars) - n_active)/(2**0.5*sigma))
    
    # Initiate lists to hold number of active covid cases
    Active = np.zeros(len(fdates))
    Active[0] = new_cases[0]
    
    
    # Roll up patient carry-over into lists of total critical care and total
    # non-critical patients expected
    
    for i, day in enumerate(fdates):
        
        inactive = Active * p_active
        #a1 = np.sum(inactive)
        Active = Active - inactive
        Active = np.roll(Active, shift=1)
        Active[0] = new_cases[i]
        total_active.append(np.sum(Active))
        
    for i, val in enumerate(forecasted_y):
        try:
            new_obs[i]
        except:
            new_obs.append(np.nan)
    
    cells = []
    for i in range(len(row_labels)):
            
        new = new_cases[i]
        cell = [row_labels[i], int(np.round(total_active[i])), int(np.round(new)), new_obs[i]]
        cells.append(cell)
        
    # Add the row to the dataframe
    df = pd.DataFrame.from_records(cells, columns=col_labels)    
    
    #### Construct arrays for critical care and non-critical care patients
        
    # Add the row to the dataframe
    
    df = df.to_json()
    
    fdates = 0
    forecasted_y = 0
    col_labels = 0
    cells = 0
    cell = 0
    row_labels = 0
    total_active = 0
    Active = 0
    
    return df




        
def generate_plot_patient_census(census_df, reset):
    
    census_df = pd.read_json(census_df)
    
    nogo = ['GLOVE SURGICAL', 'GLOVE EXAM NITRILE', 
            'GLOVE EXAM VINYL', 'MASK FACE PROCEDURE ANTI FOG',
            'MASK PROCEDURE FLUID RESISTANT', 'GOWN ISOLATION XLARGE YELLOW', 
            'MASK SURGICAL ANTI FOG W/FILM', 'SHIELD FACE FULL ANTI FOG',
            'RESPIRATOR PARTICULATE FILTER REG']
    
    census_df.drop(nogo, axis=1, inplace=True)
    
    labels = list(census_df)
    
    labels = labels[1:]
    fig_data = []
    
    clrs = ['', 'black', '#737373', 'black', 'purple',  'mediumorchid', 'blue', 'dodgerblue', 'deepskyblue',
            'green', 'limegreen', 'gold', 'orange', 'red', 'darkred']
    
    for i, label in enumerate(labels):
        
        if label in ['date', 'Discharged from ICU deceased', 
                     'Discharged from ICU alive',
                     'Discharged from non-ICU alive',
                     'Total cases']:
            
            continue
        
        dates = census_df['date'].tolist()
        clr = clrs[i]
        obs_y = census_df[label].tolist()

        if label != 'New cases (Observed)':
            fig_data.append(
                go.Scatter(
                    x=dates,
                    y=obs_y,
                    mode="lines",
                    name=label,
                    #visible=lo,
                    opacity=0.75,
                    line=dict(color=clr, width=2)
                )
            )
        else:
            fig_data.append(
                go.Scatter(
                    x=dates,
                    y=obs_y,
                    mode="markers",
                    name=label,
                    #visible=lo,
                    opacity=0.75,
                    marker=dict(size=10,
                                color='DarkSlateGrey'),
                )
            )
        
    
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(
                title=dict(
                    text="<b>Date</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            yaxis=dict(
                title=dict(
                    text="<b>COVID-19 patients</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                        
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            margin=dict(l=60, r=30, b=10, t=40),
            showlegend=True,
            height=400,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),
    )
    
    dates = 0
    census_df = 0
    
    return figure



def generate_plot_new_cases(df, loc, cty, reset):
    
    pop_size = int()
    if cty == 'Entire state or territory':
        
        pop_size = statepops[statepops['Province/State'] == loc]['PopSize'].tolist()
        pop_size = pop_size[0]
    
    else:
        cty_pops = pd.read_pickle('DataUpdate/data/County_Pops.pkl')
        
        try:
            pop_size = cty_pops[(cty_pops['State'] == loc) & (cty_pops['County'] == cty)]['Population size'].iloc[0]
    
        except:
            pop_size = 0
        
    cty_pops = 0
     
    df = pd.read_json(df)
    df = df[df['label'] == 'Current forecast']
    
    #df['forecast_dates'] = df['forecast_dates'].dt.strftime('%m/%d')#.values.tolist()
    fdates = df['forecast_dates'].iloc[0]
    y = df['obs_y'].iloc[0]
    forecasted_y = df['forecasted_y'].iloc[0]
    
    new_cases = []
    ForecastDays = 60
    
    # add 1 to number of forecast days for indexing purposes
    ForecastDays = int(ForecastDays+1)
    
    # designature plot label for legend
    for i, val in enumerate(forecasted_y):
        if i > 0:
            if forecasted_y[i] - forecasted_y[i-1] > 0:
                new_cases.append(forecasted_y[i] - forecasted_y[i-1])
            else:
                new_cases.append(0)
        if i == 0:
            new_cases.append(0)
            
    new_obs = []
    for i, val in enumerate(y):
        if i > 0:
            if y[i] - y[i-1] > 0:
                new_obs.append(y[i] - y[i-1])
            else:
                new_obs.append(0)
        if i == 0:
            new_obs.append(0)

                
    # declare column labels
    col_labels = ['date', 'Active cases', 'New cases (Forecasted)', 'New cases (Observed)']
    
    # row labels are the dates
    row_labels = fdates

    #### Construct arrays for critical care and non-critical care patients
    total_active = []
    
    ## Use lognormal to parameterize turnover of active cases
    x_vars = np.array(list(range(1, len(fdates)+1)))
    
    sigma = 0.1
    n_active = np.log(14) - (sigma**2)/2
    p_active = 0.5 + 0.5 * sc.special.erf((np.log(x_vars) - n_active)/(2**0.5*sigma))
    
    # Initiate lists to hold number of active covid cases
    Active = np.zeros(len(fdates))
    Active[0] = new_cases[0]
    
    
    # Roll up patient carry-over into lists of total critical care and total
    # non-critical patients expected
    
    for i, day in enumerate(fdates):
        
        inactive = Active * p_active
        #a1 = np.sum(inactive)
        Active = Active - inactive
        Active = np.roll(Active, shift=1)
        Active[0] = new_cases[i]
        total_active.append(np.sum(Active))
        
    for i, val in enumerate(forecasted_y):
        try:
            new_obs[i]
        except:
            new_obs.append(np.nan)
    
    cells = []
    for i in range(len(row_labels)):
            
        new = new_cases[i]
        cell = [row_labels[i], int(np.round(total_active[i])), int(np.round(new)), new_obs[i]]
        cells.append(cell)
        
    # Add the row to the dataframe
    df = pd.DataFrame.from_records(cells, columns=col_labels)    
    
    #### Construct arrays for critical care and non-critical care patients
        
    # Add the row to the dataframe
    
    fdates = 0
    forecasted_y = 0
    col_labels = 0
    cells = 0
    cell = 0
    row_labels = 0
    total_active = 0
    Active = 0
    
    #df = pd.read_json(df)
    labels = list(df)
    labels = labels[1:]
    fig_data = []
    
    clrs = ['#cc0000', '#2e5cb8', '#2e5cb8', 'purple',  'mediumorchid', 'blue', 'dodgerblue', 'deepskyblue',
            'green', 'limegreen', 'gold', 'orange', 'red', 'darkred']
    
    for i, label in enumerate(labels):
        #if label in ['Active cases', 'New cases (Forecasted)', 'New cases (Observed)']:
        #    lo = 'legendonly'
        #else:
        #    lo = True
            
        if label in ['date']:
            continue
        
        label2 = str(label)
        if label2 == 'Active cases':
            label2 = 'Active cases (Forecasted)'
            
        dates = df['date'].tolist()
        clr = clrs[i]
        obs_y = df[label].tolist()

        if label != 'New cases (Observed)':
            fig_data.append(
                go.Scatter(
                    x=dates,
                    y=obs_y,
                    mode="lines",
                    name=label2,
                    #visible=lo,
                    opacity=0.95,
                    line=dict(color=clr, width=2)
                )
            )
        else:
            fig_data.append(
                go.Scatter(
                    x=dates,
                    y=obs_y,
                    mode="markers",
                    name=label2,
                    #visible=lo,
                    opacity=0.65,
                    marker=dict(size=10,
                                color=clr),
                )
            )
        
    if pop_size > 0:
        p_active = 100 * df['Active cases']/pop_size
        fig_data.append(
                go.Scatter(
                    x=dates,
                    y=p_active,
                    mode="lines",
                    name='% active cases per capita',
                    #visible=lo,
                    opacity=0.75,
                    marker=dict(size=10,
                                color='red'),
                )
            )
        
    
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(
                title=dict(
                    text="<b>Date</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            yaxis=dict(
                title=dict(
                    text="<b>COVID-19 cases</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                        
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            margin=dict(l=60, r=30, b=10, t=40),
            showlegend=True,
            height=400,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),
    )
    
    dates = 0
    df = 0
    
    return figure





def generate_plot_employee_forecast1(df, loc, cty, employees, inc_rate, furlough, reset):
    
    pop_size = int()
    pop_size_e = int()
    
    if cty == 'Entire state or territory':
        
        pop_size = statepops[statepops['Province/State'] == loc]['PopSize'].tolist()
        pop_size = pop_size[0]
    
    
    else:
        cty_pops = pd.read_pickle('DataUpdate/data/County_Pops.pkl')
        
        try:
            pop_size = cty_pops[(cty_pops['State'] == loc) & (cty_pops['County'] == cty)]['Population size'].iloc[0]
    
        except:
            pop_size = statepops[statepops['Province/State'] == loc]['PopSize'].tolist()
            pop_size = pop_size[0]
    
    if employees is None or employees == 'Enter a number':
        pop_size_e = int(pop_size)
    else:
        pop_size_e = int(employees)
        
    cty_pops = 0
     
    df = pd.read_json(df)
    df = df[df['label'] == 'Current forecast']
    
    #df['forecast_dates'] = df['forecast_dates'].dt.strftime('%m/%d')#.values.tolist()
    fdates = df['forecast_dates'].iloc[0]
    y = df['obs_y'].iloc[0]
    forecasted_y = np.array(df['forecasted_y'].iloc[0])
    
    new_cases = []
    new_cases_e = []
    ForecastDays = 60
    
    # add 1 to number of forecast days for indexing purposes
    ForecastDays = int(ForecastDays+1)
    
    # designature plot label for legend
    for i, val in enumerate(forecasted_y):
        if i > 0:
            if forecasted_y[i] - forecasted_y[i-1] > 0:
                new_cases.append(forecasted_y[i] - forecasted_y[i-1])
            else:
                new_cases.append(0)
        if i == 0:
            new_cases.append(0)
    
    inc_rate = inc_rate/100
    new_cases_e = np.array(new_cases) * (pop_size_e/pop_size) * inc_rate
             
    
    # declare column labels
    col_labels = ['date', 'Active cases (gen pop)', 'Active employee cases', 'New employee cases']
    
    # row labels are the dates
    row_labels = fdates

    total_active = []
    total_active_e = []
    
    ## Use lognormal to parameterize turnover of active cases
    x_vars = np.array(list(range(1, len(fdates)+1)))
    
    # General population
    sigma = 0.1
    n_active = np.log(14) - (sigma**2)/2
    p_active = 0.5 + 0.5 * sc.special.erf((np.log(x_vars) - n_active)/(2**0.5*sigma))
    
    # Initiate lists to hold number of active covid cases
    Active = np.zeros(len(fdates))
    Active[0] = new_cases[0]
    
    # Employees
    sigma = 0.05
    n_active_e = (np.log(furlough) + 0.1) - (sigma**2)/2 # for median we include (sigma**2)/2
    p_active_e = 0.5 + 0.5 * sc.special.erf((np.log(x_vars) - n_active_e)/(2**0.5*sigma))
    
    # Initiate lists to hold number of active covid cases
    Active_e = np.zeros(len(fdates))
    Active_e[0] = new_cases_e[0]
    
    
    
    # Roll up patient carry-over into lists of total critical care and total
    # non-critical patients expected
    
    for i, day in enumerate(fdates):
        
        inactive = Active * p_active
        #a1 = np.sum(inactive)
        Active = Active - inactive
        Active = np.roll(Active, shift=1)
        Active[0] = new_cases[i]
        total_active.append(np.sum(Active))
        
        inactive_e = Active_e * p_active_e
        #a1 = np.sum(inactive)
        Active_e = Active_e - inactive_e
        Active_e = np.roll(Active_e, shift=1)
        Active_e[0] = new_cases_e[i]
        total_active_e.append(np.sum(Active_e))
        
    
    cells = []
    for i in range(len(row_labels)):
        cell = [row_labels[i], int(np.round(total_active[i])), int(np.round(total_active_e[i])), int(np.round(new_cases_e[i]))]
        cells.append(cell)
        
    # Add the row to the dataframe
    df = pd.DataFrame.from_records(cells, columns=col_labels)    
    
    #### Construct arrays for critical care and non-critical care patients
        
    # Add the row to the dataframe
    
    fdates = 0
    forecasted_y = 0
    col_labels = 0
    cells = 0
    cell = 0
    row_labels = 0
    total_active = 0
    Active = 0
    total_active_e = 0
    Active_e = 0
    new_cases_e = 0
    new_cases = 0
    
    #df = pd.read_json(df)
    labels = list(df)
    
    labels = labels[1:]
    fig_data = []
    
    clrs = ['#cc0000', '#2e5cb8', 'deepskyblue',  'mediumorchid', 'blue', 'dodgerblue', 'deepskyblue',
            'green', 'limegreen', 'gold', 'orange', 'red', 'darkred']
    
    for i, label in enumerate(labels):
        if label in ['New cases (Observed)']: continue
        
        if label == 'Active cases (gen pop)':
            lo = 'legendonly'
        else:
            lo = True
            
        if label in ['date']:
            continue
        
        label2 = str(label)
            
        dates = df['date'].tolist()
        clr = clrs[i]
        obs_y = df[label].tolist()

        if label == ['Active employee cases']:
            label = 'No. of employees with active COVID'
            
        fig_data.append(
            go.Scatter(
                x=dates,
                y=obs_y,
                mode="lines",
                name=label2,
                    visible=lo,
                opacity=0.95,
                line=dict(color=clr, width=2)
            )
        )
        
    if pop_size > 0:
        p_active = 100 * df['Active cases (gen pop)']/pop_size
        fig_data.append(
                go.Scatter(
                    x=dates,
                    y=p_active,
                    mode="lines",
                    name='% active cases per capita',
                    visible=lo,
                    opacity=0.5,
                    marker=dict(size=10,
                                color='red'),
                )
            )
        
    if pop_size_e > 0:
        p_active = 100 * df['Active employee cases']/pop_size_e
        fig_data.append(
                go.Scatter(
                    x=dates,
                    y=p_active,
                    mode="lines",
                    name='% of employees furloughed',
                    visible=lo,
                    opacity=0.9,
                    marker=dict(size=10,
                                color='red'),
                )
            )
        
    
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(
                title=dict(
                    text="<b>Date</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            yaxis=dict(
                title=dict(
                    text="<b>COVID-19 cases (# or %)</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                        
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            margin=dict(l=60, r=30, b=10, t=40),
            showlegend=True,
            height=500,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),
    )
    
    dates = 0
    df = 0
    
    
    
    return figure





def generate_plot_discharge_census(census_df, reset):
    
    census_df = pd.read_json(census_df)
    
    nogo = ['GLOVE SURGICAL', 'GLOVE EXAM NITRILE', 
            'GLOVE EXAM VINYL', 'MASK FACE PROCEDURE ANTI FOG',
            'MASK PROCEDURE FLUID RESISTANT', 'GOWN ISOLATION XLARGE YELLOW', 
            'MASK SURGICAL ANTI FOG W/FILM', 'SHIELD FACE FULL ANTI FOG',
            'RESPIRATOR PARTICULATE FILTER REG',
            'Total cases', 'New visits', ]
    
    census_df.drop(nogo, axis=1, inplace=True)
    
    
    labels = list(census_df)
    labels = labels[1:]
    fig_data = []
    
    clrs = ['purple',  'mediumorchid', 'blue', 'dodgerblue', 'deepskyblue',
            'green', 'limegreen', 'gold', 'orange', 'red', 'darkred']
    
    for i, label in enumerate(labels):
        if label in ['date']:
            continue
        if label in ['Discharged from ICU deceased', 'Discharged from ICU alive', 
                     'Discharged from non-ICU alive']:
        
            dates = census_df['date'].tolist()
            clr = clrs[i]
            obs_y = census_df[label].tolist()
            
            
            fig_data.append(
                go.Scatter(
                    x=dates,
                    y=obs_y,
                    mode="lines",
                    name=label,
                    opacity=0.75,
                    line=dict(color=clr, width=2)
                )
            )
        
    
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(
                title=dict(
                    text="<b>Date</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            yaxis=dict(
                title=dict(
                    text="<b>COVID-19 patients</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                        
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            margin=dict(l=60, r=30, b=10, t=40),
            showlegend=True,
            height=400,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),
    )
    
    dates = 0
    census_df = 0
    
    return figure



def generate_patient_census_table(census_df, reset):
    df_table = pd.read_json(census_df)
    
    nogo = ['GLOVE SURGICAL', 'GLOVE EXAM NITRILE', 
            'GLOVE EXAM VINYL', 'MASK FACE PROCEDURE ANTI FOG',
            'MASK PROCEDURE FLUID RESISTANT', 'GOWN ISOLATION XLARGE YELLOW', 
            'MASK SURGICAL ANTI FOG W/FILM', 'SHIELD FACE FULL ANTI FOG',
            'RESPIRATOR PARTICULATE FILTER REG', 'Total cases', 
            ]
    
    df_table.drop(nogo, axis=1, inplace=True)
    
    csv_string = df_table.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    
    df_table['dates'] = df_table['date'] 
    df_table['dates'] = pd.to_datetime(df_table['dates']).dt.date
    df_table = df_table[df_table['dates'] >= pd.Timestamp('today')]
    df_table.drop(['dates'], axis=1, inplace=True)
    
    figure = go.Figure(data=[go.Table(
        header=dict(values=list(df_table),
                fill_color='lavender',
                align='left', 
                height=30),
        cells=dict(values=[df_table['date'], #df_table['Total cases'],
                       #df_table['New cases'], df_table['Active cases'],
                       df_table['New visits'],
                       df_table['New admits'], df_table['All COVID'],
                       df_table['Non-ICU'], df_table['ICU'],
                       df_table['Vent'], 
                       df_table['Discharged from ICU deceased'],
                       df_table['Discharged from ICU alive'], 
                       df_table['Discharged from non-ICU alive']],
                       fill_color="rgb(245, 247, 249)",
                       align='left',
                       height=30))
        ],
        layout=go.Layout(
            margin=dict(l=10, r=10, b=10, t=10),
            showlegend=True,
            height=400,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),)
    
    df_table = 0
    
    return figure, csv_string



def generate_plot_ppe(df, reset):
    
    ppe_df = pd.read_json(df)
    
    nogo = ['Total cases', 'New visits', 'New admits',
                  'All COVID', 'Non-ICU', 'ICU', 'Vent',
                  'Discharged from ICU deceased', 'Discharged from ICU alive',
                  'Discharged from non-ICU alive']
    
    ppe_df.drop(nogo, axis=1, inplace=True)
    
    labels = list(ppe_df)
    
    labels = labels[1:]
    fig_data = []
    
    clrs = ['purple',  'mediumorchid', 'dodgerblue', 'deepskyblue',
            'green', 'limegreen', 'gold', 'orange', 'red']
    
    dates = ppe_df['date'].tolist()
    for i, label in enumerate(labels):
        
        if label in ['date']: continue
        
        clr = clrs[i]
        obs_y = ppe_df[label].tolist()

        
        fig_data.append(
            go.Scatter(
                x=dates,
                y=obs_y,
                mode="lines",
                name=label,
                opacity=0.75,
                line=dict(color=clr, width=2)
            )
        )
        
    
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(
                title=dict(
                    text="<b>Date</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            yaxis=dict(
                title=dict(
                    text="<b>COVID-19 patients</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                        
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            margin=dict(l=60, r=30, b=10, t=40),
            showlegend=True,
            height=400,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),
    )
    
    ppe_df = 0
    dates = 0
    
    return figure



def generate_ppe_table(df, reset):
    df_table = pd.read_json(df)
    
    nogo = ['Total cases', 'New visits', 'New admits',
                  'All COVID', 'Non-ICU', 'ICU', 'Vent',
                  'Discharged from ICU deceased', 'Discharged from ICU alive',
                  'Discharged from non-ICU alive']
    
    df_table.drop(nogo, axis=1, inplace=True)
    
    labels = list(df_table)
    labels = labels[1:]
    
    
    csv_string = df_table.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    
    df_table['dates'] = df_table['date'] 
    df_table['dates'] = pd.to_datetime(df_table['dates']).dt.date
    df_table = df_table[df_table['dates'] >= pd.Timestamp('today')]
    df_table.drop(['dates'], axis=1, inplace=True)
    
    figure = go.Figure(data=[go.Table(
        header=dict(values=list(df_table),
                fill_color='lavender',
                align='left', 
                height=30),
        cells=dict(values=[df_table['date'], df_table['GLOVE SURGICAL'], df_table['GLOVE EXAM NITRILE'],
                       df_table['GLOVE EXAM VINYL'], df_table['MASK FACE PROCEDURE ANTI FOG'],
                       df_table['MASK PROCEDURE FLUID RESISTANT'], df_table['GOWN ISOLATION XLARGE YELLOW'],
                       df_table['MASK SURGICAL ANTI FOG W/FILM'], df_table['SHIELD FACE FULL ANTI FOG'],
                       df_table['RESPIRATOR PARTICULATE FILTER REG']],
                       fill_color="rgb(245, 247, 249)",
                       align='left',
                       height=30))
        ],
        layout=go.Layout(
            margin=dict(l=10, r=10, b=10, t=10),
            showlegend=True,
            height=400,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),)
    
    df_table = 0
    
    return figure, csv_string

