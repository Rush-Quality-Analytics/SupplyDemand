import pandas as pd # data frame library

import time # library for time functionality
import datetime # library for date-time functionality
from random import choice, shuffle, randint # native randomization functions

import numpy as np # numerical python
from scipy import stats # scientific python statistical package
from scipy.stats import binom, poisson # binomial and poisson distribution functions
from scipy.optimize import curve_fit # optimization for fitting curves

import re # library for using regular expressions (text parsing)
import warnings # needed for suppression of unnecessary warnings
import base64 # functionality for encoding binary data to ASCII characters and decoding back to binary data

#### FUNCTIONS FOR MODELING THE SPREAD OF COVID-19 CASES


def obs_pred_rsquare(obs, pred):
    # Determines the prop of variability in a data set accounted for by a model
    # In other words, this determines the proportion of variation explained by
    # the 1:1 line in an observed-predicted plot.
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)

################ Simple growth-based statistical models

def logistic(x, a, b, c):
    # A general logistic function
    # x is observed data
    # a, b, c are optimized by scipy optimize curve fit 
    return a / (np.exp(-c * x) + b)


def get_logistic(obs_x, obs_y, ForecastDays):
    # obs_x: observed x values
    # obs_y: observd y values
    # ForecastDays: number of days ahead to extend prediction
    
    # convert obs_x to numpy array
    obs_x = np.array(obs_x) 
    # In fitting this model, assume that trailing zeros in obs_y data 
    # are not real but instead represent a lack of information
    # Otherwise, the logistic model will fail to fit
    for i, val in enumerate(obs_y):
        if val == 0:
            try:
                obs_y[i] = obs_y[i-1]
            except:
                pass
    # convert obs_y to numpy array
    obs_y = np.array(obs_y)
    
    try:
        # attempt to fit the logistic model to the observed data
        # popt: optimized model parameter values
        popt, pcov = curve_fit(logistic, obs_x, obs_y)
        # get predicted y values
        pred_y = logistic(obs_x, *popt)
        # extend x values by number of ForecastDays
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        # get corresponding forecasted y values, i.e., extend the predictions
        forecasted_y = logistic(forecasted_x, *popt)
        
        # prevent use of negative y values and
        # trailing zero-valued y values
        for i, val in enumerate(forecasted_y):
            if val <= 0:
                try:
                    obs_y[i] = obs_y[i-1]
                except:
                    pass
        # if the minimum y value is still less than zero
        # then use the polynomial model
        if np.min(forecasted_y) < 0:
            forecasted_y, forecasted_x, pred_y = get_polynomial(obs_x, obs_y, ForecastDays)
    except:
        # if the logistic model totally fails to fit
        # then use the polynomial model
        forecasted_y, forecasted_x, pred_y = get_polynomial(obs_x, obs_y, ForecastDays)
    
    # return the forecasted x-y values and predicted y values
    return forecasted_y, forecasted_x, pred_y


def get_exponential(obs_x, obs_y, ForecastDays):
    # obs_x: observed x values
    # obs_y: observd y values
    # ForecastDays: number of days ahead to extend prediction
    
    # convert obs_x to numpy array
    obs_x = np.array(obs_x)
    
    # In fitting this model, assume that trailing zeros in obs_y data 
    # are not real but instead represent a lack of information
    # Otherwise, the logistic model will fail to fit
    for i, val in enumerate(obs_y):
        if val == 0:
            try:
                obs_y[i] = obs_y[i-1]
            except:
                pass       
    
    # use linear regression to obtain the estimated parameters for
    # an exponential fit. This is not an optimization procedure,
    # but should not fail to provide a best fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(obs_x, np.log(obs_y))
    # convert obs_y to numpy array
    obs_y = np.array(obs_y)
    
    # get predicted y values
    pred_y = np.exp(intercept + slope*obs_x)
    # extend x values by number of ForecastDays
    forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
    # get corresponding forecasted y values, i.e., extend the predictions
    forecasted_y = np.exp(intercept + slope*forecasted_x)
    
    # return the forecasted x-y values and predicted y values
    return forecasted_y, forecasted_x, pred_y
        


def get_polynomial(obs_x, obs_y, ForecastDays):
    # obs_x: observed x values
    # obs_y: observd y values
    # ForecastDays: number of days ahead to extend prediction
    
    # convert obs_x to numpy array
    obs_x = np.array(obs_x)
    
    # In fitting this model, assume that trailing zeros in obs_y data 
    # are not real but instead represent a lack of information
    # Otherwise, the logistic model will fail to fit
    for i, val in enumerate(obs_y):
        if val == 0:
            try:
                obs_y[i] = obs_y[i-1]
            except:
                pass       
    
    # convert obs_y to numpy array
    obs_y = np.array(obs_y)
    
    try:
        # attempt to fit the polynomial model to the observed data
        # z: best-fit model parameter values
        z = np.polyfit(obs_x, obs_y, 2)
        # p: one-dimensional polynomial class; constructs the polynomial
        p = np.poly1d(z)
        # get predicted y values
        pred_y = p(obs_x)
        
        # extend x values by number of ForecastDays
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        # get corresponding forecasted y values, i.e., extend the predictions
        forecasted_y = p(forecasted_x)
    except:
        # if the polynomial model fails, the lack of a substitute here
        # will throw an error
        pass
    
    # return the forecasted x-y values and predicted y values
    return forecasted_y, forecasted_x, pred_y



def fit_curve(obs_x, obs_y, model, ForecastDays, N, T0, incubation_period, infectious_period, rho, socdist):
    # A function to fit various models to observed COVID-19 cases data according to:
    # obs_x: observed x values
    # obs_y: observed y values
    # model: the model to fit
    # ForecastDays: number of days ahead to extend predictions
    # N: population size of interest
    # T0: likely data of first infection (used by SEIR-SD model)
    # incubation_period: disease-specific epidemilogical parameter
        # average number of days until an exposed person becomes
        # begins to exhibit symptoms of infection
    # infectious_period: disease-specific epidemilogical parameter
        # average number of days an infected person is infected
    # rho: disease-specific epidemilogical parameter
        # aka basic reproductive number
        # average number of secondary infections produced by a typical case 
        # of an infection in a population where everyone is susceptible
    # socdist: population-specific social-distancing parameter
        # 
    
    # use the number of y observations as the number of x observations
    obs_x = list(range(len(obs_y)))
    # convert y and x observations to numpy arrays
    obs_x = np.array(obs_x)
    obs_y = np.array(obs_y)
    
    # Get the forecasted values, predicted values, and observed vs predicted r-square
    # value for the chosen model
    if model == 'logistic':
        forecasted_y, forecasted_x, pred_y = get_logistic(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
    elif model == 'exponential':
        forecasted_y, forecasted_x, pred_y = get_exponential(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
    elif model == 'polynomial':
        forecasted_y, forecasted_x, pred_y = get_polynomial(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
        
    elif model == 'SEIR-SD':
        # This model was inspired by: 
        # https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296
        # But it follows the traditional SEIR formulation while allowing for effects of social distancing,
        # testing delays, and likely date of first infection
        
        # Get formatted date parameters
        today = pd.to_datetime('today', format='%Y/%m/%d')
        if T0 is None:
            # If a date wasn't chosen, then use the date of Chicago's first infection as a default
            d1 = pd.to_datetime('2020/01/26', format='%Y/%m/%d')
        else:
            # Otherwise, use TO
            d1 = pd.to_datetime(T0, format='%Y-%m-%d')
        
        # number of days between TO and the end of the forecast window
        t_max = (today-d1).days + ForecastDays
        
        # Initial SEIR parameters
        S = [1 - 1/N] # fraction susceptible
        E = [1/N] # fraction exposed
        I = [0] # fraction infected
        R = [0] # fraction recovered
        init_vals = S, E, I, R

        #### 3 parameters of the classic SEIR model: alpha, beta, gamma
        # inverse of the incubation period
        alpha = 1/incubation_period 
        # inverse of the infectious period
        gamma = 1/infectious_period
        # contact rate, derived from alpha & gamma
            # Suppose infectious individuals make an average of x1 = beta infection-producing 
            # contacts per unit time, with a mean infectious period of x2 = 1/gamma. 
            # Then the basic reproduction number (rho) is: 
            # rho = x1 * x2
            #     = beta * 1/gamma
            #     = beta/gamma
            # And, beta = rho*gamma
        beta = rho*gamma
        params = alpha, beta, gamma
        
        # a list of time steps to iterate: integers ranging from 0 to t_max-1
        t = list(range(t_max))
        
        # Run SEIR-SD model and get forecasted and predicted values
        forecasted_y, forecasted_x, pred_y = seir_sd(obs_x, obs_y, ForecastDays,
                                        init_vals, params, N, t, socdist)
        
        # Get r-square for observed vs. predicted values
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
        
    # return the r-square and observed, predicted, and forecasted values
    return obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y



#################### Epidemiological models
def correct_beta(socdist, beta, percent_infected):
    # A function to adjust the contact rate (beta) by the percent infected
    
    # Convert the user-entered social distancing value
    # to a small decimal number (sd).
    sd = 0.0000000001 * (105 - socdist)
    # The small magnitude of sd allows the function below to:
    #    1. Adjust beta between 0 and 100% of its given value
    #    2. Make significant changes to beta before the percent 
    #       infected increases to a substantial fraction of the 
    #       population size. In short, social distancing policies
    #       have been implemented before 1% (or even 0.001%) of 
    #       the population has been infected
    
    beta = beta * (sd/(sd + percent_infected))
    return beta

def test_effect(i):
    # A logistic function with an output ranging between 0 and 1 
    # 'i' is the time step
    # This function corrects the apparent number of infected 
    # according to an assumption that testing for COVID-19 was
    # minimal in the first few weeks of infection
    return 1/(1+np.exp(-0.1*i+5))
    

def seir_sd(obs_x, obs_y, ForecastDays, init_vals, params, N, t, socdist):
    # A function to fit various models to observed COVID-19 cases data according to:
        # obs_x: observed x values
        # obs_y: observed y values
        # model: the model to fit
        # ForecastDays: number of days ahead to extend predictions
        # N: population size of interest
        # T0: likely data of first infection (used by SEIR-SD model)
        # incubation_period: disease-specific epidemilogical parameter
            # average number of days until an exposed person becomes
            # begins to exhibit symptoms of infection
        # infectious_period: disease-specific epidemilogical parameter
            # average number of days an infected person is infected
        # rho: disease-specific epidemilogical parameter
            # aka basic reproductive number
            # average number of secondary infections produced by a typical case 
            # of an infection in a population where everyone is susceptible
        # socdist: population-specific social-distancing parameter
        
    # This model was inspired by: 
    # https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296
    # But it follows the traditional SEIR formulation while allowing for effects of social distancing,
    # testing delays, and likely date of first infection
    
    # Assume that trailing zeros in obs_y data are not real but instead 
    # represent a lack of information.
    obs_x = np.array(obs_x)
    for i, val in enumerate(obs_y):
        if val == 0:
            try:
                obs_y[i] = obs_y[i-1]
            except:
                pass
    
    # unpack initial values
    S, E, I, R = init_vals
    
    # declare a list that will hold testing-corrected
    # number of infections
    Ir = list(I)
    
    # unpack parameters
    alpha, beta, gamma = params
    
    # loop through time steps from date of first likely infection
    # to end of forecast window
    for i in t[1:]:
        
        # fraction infected is the last element in the I list
        I_N = I[-1]/N
        # adjust the contact rate (beta) by the % infected
        beta = correct_beta(socdist, beta, I_N)
        
        # No. susceptible at time t = S - beta*S*I
        next_S = S[-1] - beta*S[-1]*I[-1]
        
        # No. exposed at time t = S - beta*S*I - alpha*E
        next_E = E[-1] + beta*S[-1]*I[-1] - alpha*E[-1]
        
        # No. infected at time t = I + alpha*E - gamma*I
        next_I = I[-1] + alpha*E[-1] - gamma*I[-1]
        
        # No. recovered at time t = R - gamma*I
        next_R = R[-1] + (gamma*I[-1])
        
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
        
        # get the testing lag
        test_lag = test_effect(i)
        # get apparent infected by multiplying I by the testing lag
        Ir.append(next_I * test_lag)
    
    # multiply array of apparent percent infected by N
    I = np.array(Ir)*N
    # convert I to a list
    I = I.tolist()
    
    # number of observed days + size of forecast window
    num = len(obs_x)+ForecastDays
    
    # trim extra days, i.e., between T0 and day of first observation
    forecasted_y = I[-num+1:]
    
    # forecasted_x is simply a list of values starting from 0
    forecasted_x = range(len(forecasted_y))
    # pred_y is a smaller list than forecasted_y
    pred_y = forecasted_y[:-ForecastDays+1]
    
    # predicted, and forecasted values
    return forecasted_y, forecasted_x, pred_y
