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



def fit_curve(obs_x, obs_y, model, ForecastDays, N, ArrivalDate):
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
        
    elif model == 'SEIR-SD (Requires 1-3 minutes to optimize)':        
        forecasted_y, forecasted_x, pred_y = get_seir(obs_x, obs_y, ForecastDays, N)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
        
    # return the r-square and observed, predicted, and forecasted values
    return obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y









def get_seir(obs_x, obs_y, ForecastDays, N):
    
    
    def correct_beta(sd, beta, fraction_infected):
        # A function to adjust the contact rate (beta) by the percent infected
        
        pi = fraction_infected
        beta = beta/(sd*pi + 1)
        
        return beta



    def test_effect(i):
        # A logistic function with an output ranging between 0 and 1 
        # 'i' is the time step
        # This function corrects the apparent number of infected 
        # according to an assumption that testing for COVID-19 was
        # minimal in the first few weeks of infection
        
        return 1/(1+np.exp(-0.1*i+5.8))
    
    
    def seir(obs_pred_r2x, alpha, beta, gamma, t_max, sd, fdays=0):
        
        t_max += fdays
        t = list(range(int(t_max))) 
        
        # unpack initial values
        S = [1 - 1/N] # fraction susceptible
        E = [1/N] # fraction exposed
        I = [0] # fraction infected
        R = [0] # fraction recovered
        
        # declare a list that will hold testing-corrected
        # number of infections
        Ir = list(I)
        
        # loop through time steps from date of first likely infection
        
        # to end of forecast window
        
        for i in t[1:]:
            
            #print('hi', t_max, alpha, beta, gamma)
            
            # fraction infected is the last element in the I list
    
            # adjust the contact rate (beta) by the % infected
            beta = correct_beta(sd, beta, I[-1])
            
            # No. susceptible at time t = S - beta*S*I
            next_S = S[-1] - beta *S[-1]*I[-1]
            
            # No. exposed at time t = S - beta*S*I - alpha*E
            next_E = E[-1] + beta *S[-1]*I[-1] - alpha*E[-1]
            
            # No. infected at time t = I + alpha*E - gamma*I
            next_I = I[-1] + alpha*E[-1] - gamma*I[-1] #* 0.1
            
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
        num = len(obs_x)+fdays
        
        # trim extra days, i.e., between T0 and day of first observation
        I = I[-num:]
        I = np.array(I)
        
        return I



    forecasted_y, forecasted_x, pred_y = [], [], []
    
    today = pd.to_datetime('today', format='%Y/%m/%d')
    
    sd_o = 0
    pred_y_o = []
    forecasted_y_o = []
    
    ct = 0
    while ct < 100:
        
        incubation_period = np.random.uniform(1, 10)
        infectious_period = np.random.uniform(1, 14)
        rho = np.random.uniform(1, 6)
        sd = np.random.uniform(0, 100)
        d1 = np.random.randint(1, 50)
        
        ArrivalDate = today - datetime.timedelta(days = d1 + len(obs_x))
        t_max = (today - ArrivalDate).days
        
        alpha = 1/incubation_period 
        gamma = 1/infectious_period
        beta = rho*gamma

        pred_y = seir(obs_x, alpha, beta, gamma, t_max, sd)
        
        
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
        
        
        if obs_pred_r2 > 0.99:
            ct += 1
            
            pred_y_o.append(pred_y)
            
            forecasted_y = seir(obs_x, alpha, beta, gamma, t_max, sd, ForecastDays-1)
            forecasted_y_o.append(forecasted_y)
            
            
    pred_y = [float(sum(col))/len(col) for col in zip(*pred_y_o)]
    forecasted_y = [float(sum(col))/len(col) for col in zip(*forecasted_y_o)]
    
    forecasted_x = list(range(len(forecasted_y)))
    
    # return the forecasted x-y values and predicted y values
    return forecasted_y, forecasted_x, pred_y