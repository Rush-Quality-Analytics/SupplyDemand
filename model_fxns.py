import pandas as pd # data frame library

import scipy as sc
import datetime # library for date-time functionality
import numpy as np # numerical python
from scipy import stats # scientific python statistical package
from scipy.optimize import curve_fit # optimization for fitting curves

#### FUNCTIONS FOR MODELING THE SPREAD OF COVID-19 CASES

def obs_pred_rsquare(obs, pred):
    # Determines the prop of variability in a data set accounted for by a model
    # In other words, this determines the proportion of variation explained by
    # the 1:1 line in an observed-predicted plot.
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)

################ Simple growth-based statistical models


def get_gaussian(obs_x, obs_y, ForecastDays):
    
    def gaussian1(x, n, s, m):  
        #return n**2 * (1/(s*((2*pi)**0.5))) * np.exp(-0.5 * ((x - m)/s)**2)
        return n**2 * 0.5 * (1 + sc.special.erf((x - m)/(s*2**0.5)))
    
    def gaussian2(x, n, s, m, n1, s1, m1):
        return n**2 * 0.5 * ((1 + sc.special.erf((x - m)/(s*2**0.5))) + (1 + sc.special.erf((x - m1)/(s1*2**0.5))))
    
    def gaussian3(x, n, s, m, n1, s1, m1, n2, s2, m2):  
        return n**2 * 0.5 * ((1 + sc.special.erf((x - m)/(s*2**0.5))) + (1 + sc.special.erf((x - m1)/(s1*2**0.5))) + (1 + sc.special.erf((x - m2)/(s2*2**0.5))))
    
    # obs_x: observed x values
    # obs_y: observd y values
    # ForecastDays: number of days ahead to extend prediction
    
    # convert obs_x to numpy array
    obs_x = np.array(obs_x) 
    # In fitting this model, assume that trailing zeros in obs_y data 
    # are not real but instead represent a lack of information
    # Otherwise, the logistic model will fail to fit
    
    # convert obs_y to numpy array
    obs_y = np.array(obs_y)
    
    try:
        # attempt to fit the logistic model to the observed data
        # popt: optimized model parameter values
        popt, pcov = curve_fit(gaussian3, obs_x, obs_y,
                               method='lm', maxfev=20000)
        # get predicted y values
        pred_y = gaussian3(obs_x, *popt)
        # extend x values by number of ForecastDays
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        # get corresponding forecasted y values, i.e., extend the predictions
        forecasted_y = gaussian3(forecasted_x, *popt)
        
        
        
    except:
        try:
            
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            popt, pcov = curve_fit(gaussian2, obs_x, obs_y,
                                   method='lm', maxfev=20000)
            # get predicted y values
            pred_y = gaussian2(obs_x, *popt)
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = gaussian2(forecasted_x, *popt)
            
        
        except:
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            popt, pcov = curve_fit(gaussian1, obs_x, obs_y,
                                   method='lm', maxfev=20000)
            # get predicted y values
            pred_y = gaussian1(obs_x, *popt)
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = gaussian1(forecasted_x, *popt)
    
    # return the forecasted x-y values and predicted y values
    params = []
    
    for i, val in enumerate(forecasted_y):
        if val > 5*max(pred_y):
            forecasted_y[i] = 5*max(pred_y)
            
        elif i > 0 and val <= 0 and forecasted_y[i-1] > 0:
            forecasted_y[i] = max(pred_y)
            
        if i > 0 and forecasted_y[i] < forecasted_y[i-1]:
            forecasted_y[i] = forecasted_y[i-1]
            
    return forecasted_y, forecasted_x, pred_y, params




def get_logistic(obs_x, obs_y, ForecastDays):
    
    def logistic1(x, a, b, c, d, g, f):
        # A general logistic function
        # x is observed data
        # a, b, c are optimized by scipy optimize curve fit
        #return a / (d + np.exp(-c * (x + g*np.sin(f*x)) + b))
        return a / (d + np.exp(-c * x + b))
    
    def logistic2(x, a, b, c, d, g, f, a1, b1, c1, d1):
        return a / (d + np.exp(-c * x + b)) + a1 / (d1 + np.exp(-c1 * x + b1))
    
    def logistic3(x, a, b, c, d, a1, b1, c1, d1,  a2, b2, c2, d2):
        return a / (d + np.exp(-c * x + b))   +   a1 / (d1 + np.exp(-c1 * x + b1))   +   a2 / (d2 + np.exp(-c2 * x + b2))
        
    
    # obs_x: observed x values
    # obs_y: observd y values
    # ForecastDays: number of days ahead to extend prediction
    
    # In fitting this model, assume that trailing zeros in obs_y data 
    # are not real but instead represent a lack of information
    # Otherwise, the logistic model will fail to fit
    for i, val in enumerate(obs_y):
        if val == 0:
            try:
                obs_y[i] = obs_y[i-1]
            except:
                pass
    
    try:
        
        # attempt to fit the logistic model to the observed data
        # popt: optimized model parameter values
        popt, pcov = curve_fit(logistic3, 
                               np.float64(obs_x), 
                               np.float64(obs_y), 
                               method='lm', maxfev=40000)
        
        
        pred_y = logistic3(np.float64(obs_x), *popt)
        # extend x values by number of ForecastDays
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        # get corresponding forecasted y values, i.e., extend the predictions
        forecasted_y = logistic3(np.float64(forecasted_x), *popt)
        
        
    except:
        
        try:
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            popt, pcov = curve_fit(logistic2, 
                                   np.float64(obs_x), 
                                   np.float64(obs_y), 
                                   method='lm', maxfev=40000)
            
            
            pred_y = logistic2(np.float64(obs_x), *popt)
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = logistic2(np.float64(forecasted_x), *popt)
            
                
        except:
            
            # attempt to fit the logistic model to the observed data
            # popt: optimized model parameter values
            popt, pcov = curve_fit(logistic1, 
                                   np.float64(obs_x), 
                                   np.float64(obs_y), 
                                   method='lm', maxfev=40000)
            
            
            pred_y = logistic1(np.float64(obs_x), *popt)
            # extend x values by number of ForecastDays
            forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
            # get corresponding forecasted y values, i.e., extend the predictions
            forecasted_y = logistic1(np.float64(forecasted_x), *popt)
    
    params = []
    
    for i, val in enumerate(forecasted_y):
        if val > 5*max(pred_y):
            forecasted_y[i] = 5*max(pred_y)
            
        elif i > 0 and val <= 0 and forecasted_y[i-1] > 0:
            forecasted_y[i] = max(pred_y)
            
        if i > 0 and forecasted_y[i] < forecasted_y[i-1]:
            forecasted_y[i] = forecasted_y[i-1]
        
    
    return forecasted_y, forecasted_x, pred_y, params





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
    # convert obs_y to numpy array
    obs_y = obs_y.tolist()      
    
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
    params = []
    
    obs_x = 0
    obs_y = 0
    
    fy = []
    for i, val in enumerate(forecasted_y):
        if val > 5*max(pred_y):
            fy.append(5*max(pred_y))
            
        else:
            fy.append(val)
            
            
    return fy, forecasted_x, pred_y, params
        


def get_polynomial(obs_x, obs_y, ForecastDays, degree=2):
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
    obs_y = obs_y.tolist()
    
    try:
        # attempt to fit the polynomial model to the observed data
        # z: best-fit model parameter values
        z = np.polyfit(obs_x, obs_y, degree)
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
    
    params = []
    obs_x = 0
    obs_y = 0
    
    fy = []
    for i, val in enumerate(forecasted_y):
        if val > 5*max(pred_y):
            fy.append(5*max(pred_y))
            
        else:
            fy.append(val)
            
    # return the forecasted x-y values and predicted y values
    return fy, forecasted_x, pred_y, params




def fit_curve(obs_x, obs_y, model, ForecastDays, N, ArrivalDate, day, iterations):
    
    # A function to fit various models to observed COVID-19 cases data according to:
    # obs_x: observed x values
    # obs_y: observed y values
    # model: the model to fit
    # ForecastDays: number of days ahead to extend predictions
    # N: population size of interest

    
    # use the number of y observations as the number of x observations
    obs_x = list(range(len(obs_y)))
    # convert y and x observations to numpy arrays
    obs_x = np.array(obs_x)
    obs_y = np.array(obs_y)
    
    
    # Get the forecasted values, predicted values, and observed vs predicted r-square
    # value for the chosen model
    
    if model == 'Logistic (multi-phase)':
        forecasted_y, forecasted_x, pred_y, params = get_logistic(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
        
    elif model == 'Gaussian (multi-phase)':
        forecasted_y, forecasted_x, pred_y, params = get_gaussian(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
    
    elif model == 'Exponential':
        forecasted_y, forecasted_x, pred_y, params = get_exponential(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
    
    elif model == 'Quadratic':
        forecasted_y, forecasted_x, pred_y, params = get_polynomial(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y[-30:], pred_y[-30:])
        
    obs_y = 0
    
    return obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y, params


