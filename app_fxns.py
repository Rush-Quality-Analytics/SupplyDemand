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

import model_fxns as fxns



testing_df_mrd = pd.read_pickle('DataUpdate/data/Testing_Dataframe_Most_Recent_Day.pkl')
testing_df = pd.read_pickle('DataUpdate/data/Testing_Dataframe.pkl')

col_names1 =  ['obs_y', 'pred_y', 'forecasted_y', 'pred_dates', 'label', 
               'forecast_dates',  'obs_pred_r2', 'model', 'focal_loc', 
               'PopSize', 'ArrivalDate', 'pred_clr', 'fore_clr']
fits_df = pd.DataFrame(columns = col_names1)


col_names2 = ['Total cases', 'New cases', 'New visits', 'New admits',
                  'All COVID', 'Non-ICU', 'ICU', 'Vent',
                  'Discharged from ICU deceased', 'Discharged from ICU alive',
                  'Discharged from non-ICU alive']
census_df = pd.DataFrame(columns = col_names2)

seir_fits_df = pd.read_csv('DataUpdate/data/SEIR-SD_States_Update.txt', sep='\t')
statepops = pd.read_csv('DataUpdate/data/StatePops.csv')

locs_df = pd.read_csv('DataUpdate/data/COVID-CASES-DF.txt', sep='\t') 
locs_df = locs_df[locs_df['Country/Region'] == 'US']
locs_df = locs_df[~locs_df['Province/State'].isin(['US', 'American Samoa', 'Northern Mariana Islands',
                                                'Diamond Princess', 'Grand Princess', 'Recovered', 
                                                 'United States Virgin Islands', 'Virgin Islands, U.S.',
                                                'Wuhan Evacuee'])]

locs_df.drop(columns=['Unnamed: 0'], inplace=True)

locations = list(set(locs_df['Province/State']))
locations.sort()

models = ['SEIR-SD', '2 phase sine-logistic', '2 phase logistic', 'Logistic', 'Gaussian', 'Quadratic', 'Exponential']
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
            html.H5("Instructions for using the COVID Calculator", style={
            'textAlign': 'left',
            }),
            dcc.Markdown("-------"),
            dcc.Markdown("**MODEL FORECASTS**"),
            dcc.Markdown("**1. Select a State and a Model to fit.**" +
                         " Choose from 7 models to obtain forecasts for COVID-19 cases across US states and select terroritories." + 
                         " See the [preprint](https://www.medrxiv.org/content/10.1101/2020.04.20.20073031v2) or Details box below for an explanation of each model." + 
                         " Some models run very quickly (exponential, logistic, quadratic, gaussian, SEIR-SD)." + 
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
            dcc.Markdown("**See the Details box below or our [preprint](https://www.medrxiv.org/content/10.1101/2020.04.20.20073031v2) for deeper insights.**"),
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
            dcc.Markdown("**Logistic:** " +  
                         "Exponential growth within a population cannot continue *ad infinitum*. Instead, growth must slow as an upper limit " +
                         "is approached or as natural limitations to disease spread (e.g., immunity, contact among hosts) are encountered. " +
                         "The logistic model captures this slowing and eventual saturation, resulting in a sigmoidal or s-shaped growth curve. " +
                         "In addition to exponential and quadratic growth, early COVID-19 studies have implicated logistic growth in the spread " + 
                         "of the disease. The logistic model takes a relatively simple functional form, " +
                         "N_t=α/(1+e^(-rt) ), where α is the upper limit of N and r is the intrinsic rate of increase. Our application uses " +
                         "numerical optimization of α and r to find the best fit logistic function and hence, predicted values for N. "
                         ),
            html.Br(),
            dcc.Markdown("**Gaussian:** " +  
                         "The Gaussian (i.e., normal) distribution can provide a relatively simple and close approximation to complex epidemiological " +
                         "models. This symmetrical curve has two parameters, mean = μ, standard deviation = σ, and belongs to the family of exponential " +
                         "distributions. When used to model spread of disease, Gaussian curves are symmetrical around a climax day with the change " +
                         "in the rate of growth determining the standard deviation about the curve. Gaussian models have previously been successful " +
                         "in approximating the spread of COVID-19 in Germany. Our application uses numerical optimization of μ and σ and the " +
                         "cumulative distribution function of the Gaussian model to find the best fit cumulative Gaussian function and hence, " +
                         "predicted values for N."
                         ),
            html.Br(),
            dcc.Markdown("**SEIR-SD:** " +  
                         "To date, COVID-19 studies have used a variety of epidemiological models to characterize the spread of the disease within " +
                         "populations. The modeling in several of these studies has been based on refinements to the classic SEIR model. In this model, " +
                         "a contagious disease drives changes in the fraction of susceptible persons (S), the fraction of persons exposed but not yet " +
                         "exhibiting infection (E), the fraction of infectious persons (I), and the fraction of persons recovered (R), where S + E + I + R = 1. "+
                         "These SEIR subpopulations are modeled as compartments in a set of ordinary differential equations:"
                         ),
            dcc.Markdown("dS/dt = βSI , dE/dt = βSI-αE , dI/dt = αE-γI , dR/dt = γI"),
            dcc.Markdown("In these equations, α is the inverse of the incubation period, and γ is the inverse of the average infectious period, and β is " +
                         "the average number of contacts of infected persons with susceptible persons per unit time. Our application imputes the initial " +
                         "value of β from a well-known simplifying relationship between γ and the basic reproductive number (R0), i.e., β = γ R0 [20-22]." +
                         "We allowed β to decrease in proportion to I. We assumed that people will, on average, reduce their contact with others when the " + 
                         "populace is aware that an increasing percent of their population is infected. This approach allows an inherent degree of social " +
                         "distancing to emerge as a frequency-dependent phenomenon. We also simulated an explicit effect of social distancing (λ) to capture " +
                         "the overall strength of response to public health policies. These effects were included as time-iterative modifications to β." +
                         "We also modified the classic SEIR model to account for initial time lags in COVID-19 testing. Specifically, and particularly " +
                         "in the US, widespread testing for COVID-19 may have artificially dampened the apparent number of positive cases within the first "+
                         "month of the first reported infection. We accounted for this effect by modifying the apparent size of I while allowing the actual "+
                         "size of I to grow according to the SEIR-SD dynamic. " +
                         "This modification models testing as low-to-nonexistent during the initial weeks of outbreak, and then accelerates afterwards. " +
                         "Our application performs a pseudo-optimization on the SEIR-SD model parameters and a likely date of initial " +
                         "infection, as opposed to using the first reported occurrence. Our implementation of the SEIR-SD model is based on an unbiased " +
                         "search of multivariate parameter space within ranges of parameter values derived from population sizes for US states and territories "+
                         "and the increasing corpus of COVID-19 literature. Our application performs 200,000 iterations and chooses the set of parameters "+
                         "that maximize the explained variation in observed data. This implementation avoids the computational challenges of applying "+
                         "numerical optimizers to complex simulation models and avoids the problems that these optimizers can have in becoming trapped "+
                         "in local minima."
                         ),
            html.Br(),
            dcc.Markdown("**Resurgence model: 2-phase logistic:** " +  
                         "This model assumes that growth occurs in two primary logistic phases. That is, that growth increases exponentially and saturates "+
                         ", but then increases exponentially once more and then saturates again. Our implementation finds the optimal breakpoint between the 2 "+
                         "phases by iterating across the time series of observed growth."
                         ),
            html.Br(),
            dcc.Markdown("**Resurgence model: 2-phase sine-logistic:** " +  
                         "This model assumes that growth occurs as in the 2-phase logistic model but that growth is also characterized by periodic fluctuation, "+
                         "as in a sine-wave, hence, sine-logistic. Our implementation of the 2 phase sine-logistic model finds the optimal breakpoint between the 2 "+
                         "primary phases of growth by iterating across the time series of observed growth."
                         ),
                        
            dcc.Markdown("**See our [preprint](https://www.medrxiv.org/content/10.1101/2020.04.20.20073031v2) for deeper insights.**"),
            
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
            html.P("Select a location"),
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
            html.P("Select a model"),
            dcc.Dropdown(
                id="model-select1",
                options=[{"label": i, "value": i} for i in models],
                value='SEIR-SD',
            ),
            
            
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











def generate_model_forecasts(loc,  model, reset):
    
    new_cases = []
    ForecastDays = 60
    
    col_names =  ['obs_y', 'pred_y', 'forecasted_y', 'pred_dates', 'forecast_dates', 
                  'label', 'obs_pred_r2', 'model', 'focal_loc', 'PopSize', 
                  'ArrivalDate', 'pred_clr', 'fore_clr']
        
    fits_df = pd.DataFrame(columns = col_names)

    PopSize = statepops[statepops['Province/State'] == loc]['PopSize'].tolist()
    PopSize = PopSize[0]
        
    ArrivalDate = statepops[statepops['Province/State'] == loc]['Date_of_first_reported_infection'].tolist()
    ArrivalDate = ArrivalDate[0]
        
    SEIR_Fit = seir_fits_df[seir_fits_df['focal_loc'] == loc]
        
        
    # add 1 to number of forecast days for indexing purposes
    ForecastDays = int(ForecastDays+1)
        
        
    # filter main dataframe to include only the chosen location
    df_sub = locs_df[locs_df['Province/State'] == loc]
        
    # get column labels, will filter below to extract dates
    yi = list(df_sub)
        
        
    obs_y_trunc = []
    fore_clrs =  ['purple',  'mediumorchid', 'plum', 'blue', 'deepskyblue', 
                  'darkturquoise', 'green', 'limegreen', 'gold', 'orange', 'red']
    pred_clrs = ['0.0', '0.1', '0.2', '0.25', '0.3', '0.35', '0.4', '0.5',
                 '0.6', '0.7', '0.8']
            
    for i, j in enumerate(list(range(-10, 1))):
        pred_clr = pred_clrs[i]
        fore_clr = fore_clrs[i]
            
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
            
    
        # declare x as a list of integers from 0 to len(y)
        x = list(range(len(y)))

        # Call function to use chosen model to obtain:
        #    r-square for observed vs. predicted
        #    predicted y-values
        #    forecasted x and y values
        iterations = 2
        obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y, params = fxns.fit_curve(x, y, 
                            model, ForecastDays, PopSize, ArrivalDate, j, iterations, SEIR_Fit)
            
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
        fdates = fdates.strftime('%m/%d')
            
        # designature plot label for legend
        if j == 0:
            label='Current forecast'
            
        else:
            label = str(-j)+' day old forecast'
            
            
        if label == 'Current forecast':
            for i, val in enumerate(forecasted_y):
                if i > 0:
                    if forecasted_y[i] - forecasted_y[i-1] > 0:
                        new_cases.append(forecasted_y[i] - forecasted_y[i-1])
                    else:
                        new_cases.append(0)
                if i == 0:
                    new_cases.append(forecasted_y[i])
                        
                
        # get dates from ArrivalDate to the current day
        dates = pd.date_range(start=first_date, end=latest_date)
        dates = dates.strftime('%m/%d')
            
            
        output_list = [y.tolist(), pred_y.tolist(), forecasted_y.tolist(), dates, fdates,
                       label, obs_pred_r2, model, loc, PopSize, 
                       ArrivalDate, pred_clr, fore_clr]
            
        fits_df.loc[len(fits_df)] = output_list

    fits_df = fits_df.to_dict()
    return fits_df



    

def generate_model_forecast_plot(fits_df, reset):
    fits_df = pd.DataFrame.from_dict(fits_df)
    
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
                    text="<b>Cumulative cases</b>",
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
        title=dict(text="r<sup>2</sup> values pertain to the fits of models (colored lines) to observed data (black dots).",
                   font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                    ),
                    ),
        )
    return figure



def generate_model_forecast_table(fits_df, reset):
    df = pd.DataFrame.from_dict(fits_df)
    df_table = pd.DataFrame()
    
    col0 = df.forecast_dates[-1]
    max_len = len(col0)
    
    col1 = df.obs_y[-1]    
    zs = [0] * (max_len - len(col1))
    col1.extend(zs)
    
    col2 = df.forecasted_y[-1]
    zs = [0] * (max_len - len(col2))
    col2.extend(zs)
    
    col3 = df.forecasted_y[-2]
    zs = [0] * (max_len - len(col3))
    col3.extend(zs)
    
    col4 = df.forecasted_y[-3]
    zs = [0] * (max_len - len(col4))
    col4.extend(zs)
    
    col5 = df.forecasted_y[-4]
    zs = [0] * (max_len - len(col5))
    col5.extend(zs)
    
    col6 = df.forecasted_y[-5]
    zs = [0] * (max_len - len(col6))
    col6.extend(zs)
    
    col7 = df.forecasted_y[-6]
    zs = [0] * (max_len - len(col7))
    col7.extend(zs)
    
    col8 = df.forecasted_y[-7]
    zs = [0] * (max_len - len(col8))
    col8.extend(zs)
    
    col9 = df.forecasted_y[-8]
    zs = [0] * (max_len - len(col9))
    col9.extend(zs)
    
    col10 = df.forecasted_y[-9]
    zs = [0] * (max_len - len(col10))
    col10.extend(zs)
    
    col11 = df.forecasted_y[-10]
    zs = [0] * (max_len - len(col11))
    col11.extend(zs)
    
    col12 = df.forecasted_y[-11]
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
    
    return csv_string
    
    
    


        
def generate_patient_census(loc,  model, icu_beds, nonicu_beds, per_loc, per_admit, 
    per_cc, LOS_cc, LOS_nc, per_vent, TimeLag, transfers, per_ICU_transfer, 
    mortality, GLOVE_SURGICAL, GLOVE_EXAM_NITRILE, GLOVE_EXAM_VINYL, 
    MASK_FACE_PROC_ANTI_FOG, MASK_PROC_FLUID_RESISTANT, GOWN_ISOLATION_XL_YELLOW, 
    MASK_SURG_ANTI_FOG_FILM, SHIELD_FACE_FULL_ANTI_FOG, RESP_PART_FILTER_REG,
    reset):
    
    new_cases = []
    ForecastDays = 60

    
    PopSize = statepops[statepops['Province/State'] == loc]['PopSize'].tolist()
    PopSize = PopSize[0]
        
    ArrivalDate = statepops[statepops['Province/State'] == loc]['Date_of_first_reported_infection'].tolist()
    ArrivalDate = ArrivalDate[0]
        
    SEIR_Fit = seir_fits_df[seir_fits_df['focal_loc'] == loc]
      
    # add 1 to number of forecast days for indexing purposes
    ForecastDays = int(ForecastDays+1)
        
        
    # filter main dataframe to include only the chosen location
    df_sub = locs_df[locs_df['Province/State'] == loc]
        
    # get column labels, will filter below to extract dates
    yi = list(df_sub)
        
    obs_y_trunc = []
    DATES = yi[4:]
    obs_y_trunc = df_sub.iloc[0,4:].values
    
    ii = 0
    while obs_y_trunc[ii] == 0: ii+=1
    y = obs_y_trunc[ii:]
    dates = DATES[ii:]
            
    
    # declare x as a list of integers from 0 to len(y)
    x = list(range(len(y)))

    # Call function to use chosen model to obtain:
    #    r-square for observed vs. predicted
    #    predicted y-values
    #    forecasted x and y values
    iterations = 2
    obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y, params = fxns.fit_curve(x, y, 
                            model, ForecastDays, PopSize, ArrivalDate, 0, iterations, SEIR_Fit)
            
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
    fdates = fdates.strftime('%m/%d')
    
    # designature plot label for legend
    for i, val in enumerate(forecasted_y):
        if i > 0:
            if forecasted_y[i] - forecasted_y[i-1] > 0:
                new_cases.append(forecasted_y[i] - forecasted_y[i-1])
            else:
                new_cases.append(0)
        if i == 0:
            new_cases.append(forecasted_y[i])
                        
                
    # get dates from ArrivalDate to the current day
    dates = pd.date_range(start=first_date, end=latest_date)
    dates = dates.strftime('%m/%d')
            
    # declare column labels
    col_labels = ['date', 'Total cases', 'New cases', 'New visits', 'New admits',
                  'All COVID', 'Non-ICU', 'ICU', 'Vent',
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

    # Declare pandas dataframe to hold data for download
    #census_df = pd.DataFrame(columns = ['date'] + col_labels)
    
    #### Construct arrays for critical care and non-critical care patients
    # use lognormal cdf
    
    sigma = 0.5
    n_cc = np.log(LOS_cc) - (sigma**2)/2
    n_nc = np.log(LOS_nc) - (sigma**2)/2
    
    x_vars = np.array(list(range(1, len(fdates)+1)))
    #y = (1/(x_vars * (2 * pi * sigma**2)**0.5)) * np.exp((-1/(2*sigma**2)) * (np.log(x_vars) - LOS_cc)**2)

    
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
            
        new = new_cases[i]
        val = ts_lag[i]
            
        # each cell is a row with 4 columns:
        #     Total cases, 
        #     new cases, 
        #     time-lagged visits to your hospital,
        #     time-lagged admits to your hospital
        
        cell = [row_labels[i],
                int(np.round(forecasted_y[i])), 
                int(np.round(new)), 
                int(np.round(val * (per_loc * 0.01))),
                int(np.round((0.01 * per_admit) * val * (per_loc * 0.01))),
                int(np.round(total_nc[i] + total_cc[i])), 
                int(np.round(total_nc[i])),
                int(np.round(total_cc[i])), 
                int(np.round(total_cc[i]*(0.01*per_vent))),
                int(discharged_dead_cc[i]),
                int(discharged_alive_cc[i]),
                int(discharged_alive_nc[i])]
        
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
    
    
    census_df = census_df.to_dict()
    
    return census_df



        
def generate_plot_patient_census(census_df, reset):
    
    census_df = pd.DataFrame.from_dict(census_df)
    
    nogo = ['GLOVE SURGICAL', 'GLOVE EXAM NITRILE', 
            'GLOVE EXAM VINYL', 'MASK FACE PROCEDURE ANTI FOG',
            'MASK PROCEDURE FLUID RESISTANT', 'GOWN ISOLATION XLARGE YELLOW', 
            'MASK SURGICAL ANTI FOG W/FILM', 'SHIELD FACE FULL ANTI FOG',
            'RESPIRATOR PARTICULATE FILTER REG']
    
    census_df.drop(nogo, axis=1, inplace=True)
    
    labels = list(census_df)
    
    labels = labels[1:]
    fig_data = []
    
    clrs = ['purple',  'mediumorchid', 'blue', 'dodgerblue', 'deepskyblue',
            'green', 'limegreen', 'gold', 'orange', 'red', 'darkred']
    
    for i, label in enumerate(labels):
        
        if label in ['date', 'Discharged from ICU deceased', 
                     'Discharged from ICU alive',
                     'Discharged from non-ICU alive',
                     'Total cases', 'New cases']:
            #, 'Total cases', 'New cases', 'New visits']:
            continue
        
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
    
    return figure






def generate_plot_discharge_census(census_df, reset):
    
    census_df = pd.DataFrame.from_dict(census_df)
    
    nogo = ['GLOVE SURGICAL', 'GLOVE EXAM NITRILE', 
            'GLOVE EXAM VINYL', 'MASK FACE PROCEDURE ANTI FOG',
            'MASK PROCEDURE FLUID RESISTANT', 'GOWN ISOLATION XLARGE YELLOW', 
            'MASK SURGICAL ANTI FOG W/FILM', 'SHIELD FACE FULL ANTI FOG',
            'RESPIRATOR PARTICULATE FILTER REG']
    
    census_df.drop(nogo, axis=1, inplace=True)
    
    
    labels = list(census_df)
    labels = labels[1:]
    fig_data = []
    
    clrs = ['purple',  'mediumorchid', 'blue', 'dodgerblue', 'deepskyblue',
            'green', 'limegreen', 'gold', 'orange', 'red', 'darkred']
    
    for i, label in enumerate(labels):
        
        if label in ['date']:#, 'Total cases', 'New cases', 'New visits']:
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
    
    return figure



def generate_patient_census_table(census_df, reset):
    df_table = pd.DataFrame.from_dict(census_df)
    
    nogo = ['GLOVE SURGICAL', 'GLOVE EXAM NITRILE', 
            'GLOVE EXAM VINYL', 'MASK FACE PROCEDURE ANTI FOG',
            'MASK PROCEDURE FLUID RESISTANT', 'GOWN ISOLATION XLARGE YELLOW', 
            'MASK SURGICAL ANTI FOG W/FILM', 'SHIELD FACE FULL ANTI FOG',
            'RESPIRATOR PARTICULATE FILTER REG']
    
    df_table.drop(nogo, axis=1, inplace=True)
    
    csv_string = df_table.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    
    df_table['dates'] = df_table['date'] + '/2020'
    df_table['dates'] = pd.to_datetime(df_table['dates']).dt.date
    df_table = df_table[df_table['dates'] >= pd.Timestamp('today')]
    df_table.drop(['dates'], axis=1, inplace=True)
    
    figure = go.Figure(data=[go.Table(
        header=dict(values=list(df_table),
                fill_color='lavender',
                align='left', 
                height=30),
        cells=dict(values=[df_table['date'], df_table['Total cases'],
                       df_table['New cases'], df_table['New visits'],
                       df_table['New admits'], df_table['All COVID'],
                       df_table['Non-ICU'], df_table['ICU'],
                       df_table['Vent'], df_table['Discharged from ICU deceased'],
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
    
    
    
    return figure, csv_string



def generate_plot_ppe(df, reset):
    
    ppe_df = pd.DataFrame.from_dict(df)
    
    nogo = ['Total cases', 'New cases', 'New visits', 'New admits',
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
    
    return figure



def generate_ppe_table(df, reset):
    df_table = pd.DataFrame.from_dict(df)
    
    nogo = ['Total cases', 'New cases', 'New visits', 'New admits',
                  'All COVID', 'Non-ICU', 'ICU', 'Vent',
                  'Discharged from ICU deceased', 'Discharged from ICU alive',
                  'Discharged from non-ICU alive']
    
    df_table.drop(nogo, axis=1, inplace=True)
    
    labels = list(df_table)
    labels = labels[1:]
    
    
    csv_string = df_table.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    
    df_table['dates'] = df_table['date'] + '/2020'
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
    
    
    
    return figure, csv_string








def map1(reset):
    
    fig = go.Figure(data=go.Choropleth(
    locations = testing_df_mrd['state'], # Spatial coordinates
    z = testing_df_mrd['Testing_Rate'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Plasma',
    colorbar_title = "Testing Rate",
    marker_line_color='grey',
    text = testing_df_mrd['date'],
    ))

    fig.update_layout(
        geo_scope='usa',
        margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True,
            height=400,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)")
    
    return fig


def map2(reset):
    
    fig = go.Figure(data=go.Choropleth(
    locations=testing_df_mrd['state'], # Spatial coordinates
    z = testing_df_mrd['Positives per capita'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Viridis',
    colorbar_title = "Positives per capita",
    marker_line_color='grey',
    text = testing_df_mrd['date'],
    ))

    fig.update_layout(
        geo_scope='usa',
        margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True,
            height=400,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)")

    return fig


def map3(reset):
    
    fig = go.Figure(data=go.Choropleth(
    locations=testing_df_mrd['state'], # Spatial coordinates
    z = testing_df_mrd['Percent positive'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Cividis',
    colorbar_title = "Percent positive",
    marker_line_color='grey',
    text = testing_df_mrd['date'],
    ))

    fig.update_layout(
        geo_scope='usa',
        margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True,
            height=400,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)")
    
    return fig


def map4(reset):
    
    fig = go.Figure(data=go.Choropleth(
    locations=testing_df_mrd['state'], # Spatial coordinates
    z = testing_df['DeltaTestingRate'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'haline',
    colorbar_title = "Change in testing rate",
    marker_line_color='grey',
    text = testing_df_mrd['date'],
    ))

    fig.update_layout(
        geo_scope='usa',
        margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True,
            height=400,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)")
    
    return fig




def map5(reset):
    
    fig = go.Figure(data=go.Choropleth(
    locations = testing_df_mrd['state'], # Spatial coordinates
    z = testing_df_mrd['hospitalizedCurrently'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'haline',
    colorbar_title = "Hospitalized",
    marker_line_color='grey',
    text = testing_df_mrd['date'],
    ))

    fig.update_layout(
        geo_scope='usa',
        margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True,
            height=400,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)")
    
    return fig




def map6(reset):
    
    
    fig = go.Figure(data=go.Choropleth(
    locations=testing_df_mrd['state'], # Spatial coordinates
    z = testing_df_mrd['inIcuCurrently'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'haline',
    colorbar_title = "In ICU",
    marker_line_color='grey',
    text = testing_df_mrd['date'],
    ))

    fig.update_layout(
        geo_scope='usa',
        margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True,
            height=400,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)")
    
    return fig




def generate_delta_testing_plot(reset):
    
    fig = px.line(testing_df, x="date", y="Testing_Rate", color="state",
              line_group="state", hover_name="state",
              labels={'Testing_Rate': 'Testing rate'})

    fig.update_xaxes(title_font=dict(size=14, family='Arial', color='black'))
    fig.update_yaxes(title_font=dict(size=14, family='Arial', color='black'))

    fig.update_layout(title_font=dict(size=14, 
                      color='black', 
                      family='Arial'),
                      showlegend=True,
                      margin=dict(l=10, r=40, b=80, t=20),
                      #height=400,
                      paper_bgcolor="rgb(245, 247, 249)",
                      plot_bgcolor="rgb(245, 247, 249)",)
    
    return fig


def generate_PopSize_vs_Tested(reset):
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(testing_df_mrd['log_PopSize'], 
                                                                   testing_df_mrd['log_People_Tested'])
    
    pred_y = slope * testing_df_mrd['log_PopSize'] + intercept
    
    lab = 'Power law slope = ' + str(np.round(slope,2)) + ', r-square = ' + str(np.round(r_value**2,2))
    
    fig = go.Figure()
    
    fig = px.scatter(testing_df_mrd, x="log_PopSize", y="log_People_Tested",
                 color='%Poor',
                 symbol='color',
                 size='%Black', hover_data=['Province_State','Confirmed', 'Deaths'],
                 labels={'log_PopSize': 'log(State population size)', 
                         'log_People_Tested': 'log(Number of tests conducted)'})
    
    fig.add_trace(go.Scatter(x=testing_df_mrd['log_PopSize'], y=pred_y,
                    mode='lines',
                    ))
    
    
    fig.update_layout(title = lab,
                      title_font=dict(size=14, 
                      color='black', 
                      family='Arial'),
                      showlegend=False,
                      margin=dict(l=10, r=10, b=80, t=20),
                      #height=400,
                      paper_bgcolor="rgb(245, 247, 249)",
                      plot_bgcolor="rgb(245, 247, 249)",)
    
    return fig


def generate_Negative_vs_Tested(reset):
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(testing_df_mrd['log_People_Tested'], testing_df_mrd['log_negative'])
    
    pred_y = slope * testing_df_mrd['log_People_Tested'] + intercept
    
    lab = 'Power law slope = ' + str(np.round(slope,2)) + ', r-square = ' + str(np.round(r_value**2,2))
    
    fig = px.scatter(testing_df_mrd, y="log_negative", x="log_People_Tested", color='%Poor',
                 symbol='color',
                 size='%Black', hover_data=['Province_State','Confirmed', 'Deaths'],
                 labels={'log_negative': 'log(Negative tests)', 
                        'log_People_Tested': 'log(Number of tests conducted)'})
    
    fig.add_trace(go.Scatter(x=testing_df_mrd['log_People_Tested'], y=pred_y,
                    mode='lines',
                    ))
    
    
    fig.update_layout(title = lab,
                      title_font=dict(size=14, 
                      color='black', 
                      family='Arial'),
                      showlegend=False,
                      margin=dict(l=10, r=40, b=80, t=20),
                      #height=400,
                      paper_bgcolor="rgb(245, 247, 249)",
                      plot_bgcolor="rgb(245, 247, 249)",)
    
    
    return fig

    

def generate_Positive_vs_Tested(reset):
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(testing_df_mrd['log_People_Tested'], testing_df_mrd['log_positive'])
    
    pred_y = slope * testing_df_mrd['log_People_Tested'] + intercept
    
    lab = 'Power law slope = ' + str(np.round(slope,2)) + ', r-square = ' + str(np.round(r_value**2,2))
    
    fig = px.scatter(testing_df_mrd, x="log_People_Tested", y="log_positive",
                     color='%Poor',
                     symbol='color',
                     #trendline="ols",
                     size='%Black', hover_data=['state','total', 'death'],
                     labels={'log_People_Tested': 'log(Number of tests conducted}', 
                             'log_positive': 'log(Positive tests)'}
                    )
    
    fig.add_trace(go.Scatter(x=testing_df_mrd['log_People_Tested'], y=pred_y,
                    mode='lines',
                    ))
    
    fig.update_layout(title = lab,
                      title_font=dict(size=14, 
                      color='black', 
                      family='Arial'),
                      showlegend=False,
                      margin=dict(l=10, r=10, b=80, t=20),
                      #height=400,
                      paper_bgcolor="rgb(245, 247, 249)",
                      plot_bgcolor="rgb(245, 247, 249)",)
    
    
    return fig



def generate_ICU_vs_Hospitalized(reset):
    
    df_sub = testing_df_mrd.filter(items=['inIcuCurrently', 'onVentilatorCurrently', 'hospitalizedCurrently',
                                '%Poor', '%Black', 'sqrt_PopSize', 'PopSize', 'state', 
                                'total', 'death', 'color'])
    df_sub.dropna(inplace=True)

    varx = df_sub['hospitalizedCurrently']
    vary = df_sub['inIcuCurrently']
    mask = ~np.isnan(varx) & ~np.isnan(vary)
    slope, intercept, r_value, p_value, std_err = stats.linregress(varx[mask], vary[mask])

    pred_y = slope * df_sub['hospitalizedCurrently'] + intercept
    
    lab = 'Slope = ' + str(np.round(slope,2)) + ', r-square = ' + str(np.round(r_value**2,2))
    
    fig = px.scatter(df_sub, x="hospitalizedCurrently", y="inIcuCurrently",
                     color='%Poor',
                     symbol='color',
                     size='%Black', hover_data=['state','total', 'death'],
                     labels={'hospitalizedCurrently': 'Number hospitalized', 
                         'inIcuCurrently': 'Patients in ICU'}
                    )
    
    fig.add_trace(go.Scatter(x=df_sub['hospitalizedCurrently'], y=pred_y,
                    mode='lines',
                    ))
    
    fig.update_layout(title = lab,
                      title_font=dict(size=14, 
                      color='black', 
                      family='Arial'),
                      showlegend=False,
                      margin=dict(l=10, r=10, b=80, t=20),
                      #height=400,
                      paper_bgcolor="rgb(245, 247, 249)",
                      plot_bgcolor="rgb(245, 247, 249)",)
    
    return fig



def generate_ventilator_vs_ICU(reset):
    
    df_sub = testing_df_mrd.filter(items=['inIcuCurrently', 'onVentilatorCurrently',
                                '%Poor', '%Black', 'sqrt_PopSize', 'PopSize', 'state', 
                                'total', 'death', 'color'])
    df_sub.dropna(inplace=True)

    varx = df_sub['inIcuCurrently']
    vary = df_sub['onVentilatorCurrently']
    mask = ~np.isnan(varx) & ~np.isnan(vary)
    slope, intercept, r_value, p_value, std_err = stats.linregress(varx[mask], vary[mask])

    pred_y = slope * df_sub['inIcuCurrently'] + intercept
    
    lab = 'Slope = ' + str(np.round(slope,2)) + ', r-square = ' + str(np.round(r_value**2,2))
    
    fig = px.scatter(df_sub, x="inIcuCurrently", y="onVentilatorCurrently",
                     color='%Poor',
                     symbol='color',
                     #trendline="ols",
                     size='%Black', hover_data=['state','total', 'death'],
                     labels={'onVentilatorCurrently': 'Patients on Ventilator', 
                         'inIcuCurrently': 'Patients in ICU'}
                    )
    
    fig.add_trace(go.Scatter(x=df_sub['inIcuCurrently'], y=pred_y,
                    mode='lines',
                    ))
    
    fig.update_layout(title = lab,
                      title_font=dict(size=14, 
                      color='black', 
                      family='Arial'),
                      showlegend=False,
                      margin=dict(l=10, r=10, b=80, t=20),
                      #height=400,
                      paper_bgcolor="rgb(245, 247, 249)",
                      plot_bgcolor="rgb(245, 247, 249)",)

    return fig