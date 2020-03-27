import numpy as np
from scipy.optimize import curve_fit
from scipy import stats




################ Simple growth-based statistical models


def logistic(x, a, b, c):
    return a / (np.exp(-c * x) + b)


def obs_pred_rsquare(obs, pred):
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)


def get_logistic(obs_x, obs_y, ForecastDays):

    obs_x = np.array(obs_x)
    for i, val in enumerate(obs_y):
        if val == 0:
            try:
                obs_y[i] = obs_y[i-1]
            except:
                pass
    
    obs_y = np.array(obs_y)
    
    try:
        popt, pcov = curve_fit(logistic, obs_x, obs_y)
        pred_y = logistic(obs_x, *popt)
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        forecasted_y = logistic(forecasted_x, *popt)
        
    except:
        print('Logistic failed to fit. Using 3rd degree polynomial.')
        forecasted_y, forecasted_x, pred_y = get_polynomial(obs_x, obs_y, ForecastDays)
        
    return forecasted_y, forecasted_x, pred_y



def get_exponential(obs_x, obs_y, ForecastDays):
    
    obs_x = np.array(obs_x)
    
    for i, val in enumerate(obs_y):
        if val == 0:
            try:
                obs_y[i] = obs_y[i-1]
            except:
                pass       
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(obs_x, np.log(obs_y))
    obs_y = np.array(obs_y)
    
    pred_y = np.exp(intercept + slope*obs_x)
    forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
    forecasted_y = np.exp(intercept + slope*forecasted_x)
    
    return forecasted_y, forecasted_x, pred_y
        


def get_polynomial(obs_x, obs_y, ForecastDays):
    
    obs_x = np.array(obs_x)
    for i, val in enumerate(obs_y):
        if val == 0:
            try:
                obs_y[i] = obs_y[i-1]
            except:
                pass       
    
    obs_y = np.array(obs_y)
    forecasted_y = np.zeros(len(obs_y))
    try:
        z = np.polyfit(obs_x, obs_y, 2)
        p = np.poly1d(z)
        pred_y = p(obs_x)
            
        forecasted_x = np.array(list(range(max(obs_x) + ForecastDays)))
        forecasted_y = p(forecasted_x)
    except:
        pass
    
    return forecasted_y, forecasted_x, pred_y



def fit_curve(obs_x, obs_y, model, df_sub, ForecastDays):

    obs_x = list(range(len(obs_y)))
    obs_x = np.array(obs_x)
    obs_y = np.array(obs_y)
    
    best_loc = str()
    
    if model == 'logistic':
        forecasted_y, forecasted_x, pred_y = get_logistic(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
    elif model == 'exponential':
        forecasted_y, forecasted_x, pred_y = get_exponential(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
    elif model == 'polynomial':
        forecasted_y, forecasted_x, pred_y = get_polynomial(obs_x, obs_y, ForecastDays)
        obs_pred_r2 = obs_pred_rsquare(obs_y, pred_y)
        
    return obs_pred_r2, model, best_loc, obs_x, pred_y, forecasted_x, forecasted_y







#################### Epidemiological models



