import pandas as pd # data frame library

import scipy as sc
import datetime # library for date-time functionality
import numpy as np # numerical python
from scipy import stats # scientific python statistical package
from scipy.optimize import curve_fit # optimization for fitting curves



def obs_pred_rsquare(obs, pred):
    # Determines the prop of variability in a data set accounted for by a model
    # In other words, this determines the proportion of variation explained by
    # the 1:1 line in an observed-predicted plot.
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)



def get_seir_sd(obs_x, obs_y, ForecastDays, N, iterations, SEIR_Fit, day):
    
    def correct_beta(sd, beta, pi):
        # A function to adjust the contact rate (beta) by the percent infected
        
        beta = beta/(sd*pi + 1)
        
        return beta



    def test_effect(i):
        # A logistic function with an output ranging between 0 and 1 
        # 'i' is the time step
        # This function corrects the apparent number of infected 
        # according to an assumption that testing for COVID-19 was
        # minimal in the first few weeks of infection
        
        return 1/(1+np.exp(-0.1*i+5.8))
    
    
    def seir(obs_x, alpha, beta, gamma, d1, sd, s, im, fdays=0):
        
        sN = int(N)
        today = pd.to_datetime('today', format='%Y/%m/%d')
        ArrivalDate = today - datetime.timedelta(days = d1 + len(obs_x))
        t_max = (today - ArrivalDate).days
        
        t_max += fdays
        t = list(range(int(t_max))) 
        
        # unpack initial values
        S = [1 - 1/sN] # fraction susceptible
        E = [1/sN] # fraction exposed
        I = [0] # fraction infected
        cI = [0]
        R = [0] # fraction recovered
        
        # declare a list that will hold testing-corrected
        # number of infections
        Ir = [0]
        
        # loop through time steps from date of first likely infection
        
        # to end of forecast window
        
        for i in t[1:]:
            
            #print('hi', t_max, alpha, beta, gamma)
            
            # fraction infected is the last element in the I list
    
            # adjust the contact rate (beta) by the % infected
            beta = correct_beta(sd, beta, I[-1])
            
            # No. susceptible at time t = S - beta*S*I
            next_S = S[-1] - beta * S[-1] * I[-1]
            
            # No. exposed at time t = S - beta*S*I - alpha*E
            next_E = E[-1] + beta * S[-1] * I[-1] - alpha * E[-1] 
            
            # No. infected at time t = I + alpha*E - gamma*I
            next_I = I[-1] + alpha * E[-1] - gamma * I[-1] 
            
            next_cI = cI[-1] + alpha * E[-1]
            
            # No. recovered at time t = R - gamma*I
            next_R = R[-1] + gamma*I[-1] #+ gamma*Q[-1]
            
            
            tS = next_S * sN
            tE = next_E * sN
            tI = next_I * sN
            tR = next_R * sN
            tcI = next_cI * sN
            
            sN = sN + im*sN
            tE = tE + im*sN
            
            next_S = tS/sN
            next_E = tE/sN
            next_I = tI/sN
            next_R = tR/sN
            next_cI = tcI/sN
            
            
            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            R.append(next_R)
            cI.append(next_cI)
            
            # get the testing lag
            test_lag = test_effect(i)
            # get apparent infected by multiplying I by the testing lag
            Ir.append(next_cI * test_lag)
        
        # multiply array of apparent percent infected by N
        I = np.array(Ir) * sN
        # convert I to a list
        I = I.tolist()
        
        
        # number of observed days + size of forecast window
        num = len(obs_x)+fdays
        
        # trim extra days, i.e., between T0 and day of first observation
        I = I[-num:]
        I = np.array(I)
        
        return I



    forecasted_y, forecasted_x, pred_y = [], [], []
    pred_y_o = []
    alpha_o, beta_o, gamma_o, d1_o, sd_o, s_o, im_o = 1,1,1,1,1,1,1
    
    if SEIR_Fit is not 0:
        ref_date = SEIR_Fit['reference_date'].iloc[day-1]
        SEIR_Fit = SEIR_Fit[SEIR_Fit['reference_date'] == ref_date]
    
    
        forecasted_y = SEIR_Fit['forecasted_y'].iloc[0]
        forecasted_y = np.fromstring(forecasted_y[1:-1], sep=' ') 
        forecasted_y = forecasted_y.tolist()
    
        pred_y_o = SEIR_Fit['pred_y'].iloc[0]
        pred_y_o = np.fromstring(pred_y_o[1:-1], sep=' ') 
        pred_y_o = pred_y_o.tolist()


        params = SEIR_Fit['params'].iloc[0]
        params = eval(params)
    
        alpha_o, beta_o, gamma_o, d1_o, sd_o, s_o, im_o = params
    
    
    incubation_period = 1/alpha_o
    infectious_period = 1/gamma_o
    rho_o = beta_o/gamma_o
    
    
    rho = float(rho_o)
    alpha = float(alpha_o)
    beta = float(beta_o)
    gamma = float(gamma_o)
    d1 = float(d1_o)
    sd = float(sd_o)
    s = float(s_o)
    im = float(im_o)
    
    r2_o = 0


    obs_x = obs_x.tolist()
    for i in range(iterations):
        
        pred_y = seir(obs_x, alpha, beta, gamma, d1, sd, s, im)   
        obs_pred_r2 = obs_pred_rsquare(obs_y[0:], pred_y[0:])
        
        if i == 0 or obs_pred_r2 >= r2_o:
            
            alpha_o = float(alpha)
            beta_o = float(beta)
            gamma_o = float(gamma)
            d1_o = float(d1)
            sd_o = float(sd)
            s_o = float(s)
            r2_o = float(obs_pred_r2)
            pred_y_o = list(pred_y)
            im_o = float(im)
            
        incubation_period = np.random.uniform(4, 6)
        infectious_period = np.random.uniform(4, 10)
        rho = np.random.uniform(1, 6)
        sd = np.random.uniform(1, 100)
        d1 = np.random.randint(1, 60)
        s = np.random.uniform(0, 1)
        im = 10**np.random.uniform(-7, -4)
        
        alpha = 1/incubation_period 
        gamma = 1/infectious_period
        beta = rho*gamma
    
    
    
    forecasted_y = seir(obs_x, alpha_o, beta_o, gamma_o, d1_o, sd_o, s_o, im_o, ForecastDays-1)
    params = [alpha_o, beta_o, gamma_o, d1_o, sd_o, s_o, im_o]
    
    # return the forecasted x-y values and predicted y values
    return forecasted_y, forecasted_x, pred_y_o, params