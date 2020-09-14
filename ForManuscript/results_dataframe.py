import pandas as pd
import sys
import numpy as np
import datetime 

import model_fxns as fxns



# IMPORT DATA
StatePops = pd.read_csv('data/StatePops.csv')
    
cases_df = pd.read_csv('data/COVID-CASES-DF.txt', sep='\t') 
cases_df = cases_df[cases_df['Country/Region'] == 'US']

cases_df = cases_df[~cases_df['Province/State'].isin(['US', 'American Samoa', 
                                        'Northern Mariana Islands', 'Diamond Princess', 
                                        'Grand Princess', 'Recovered', 
                                        'United States Virgin Islands', 
                                        'Virgin Islands, U.S.',
                                        'Wuhan Evacuee'])]


cases_df.drop(columns=['Unnamed: 0'], inplace=True)
    

models = ['SEIR-SD', '2 phase sine-logistic', '2 phase logistic',
          'Logistic', 'Exponential', 'Quadratic', 'Gaussian']

locations = list(set(cases_df['Province/State']))
locations.sort()
            
col_names =  ['pred_dates', 'obs_pred_r2', 'model', 'focal_loc']
            
model_fits_df  = pd.DataFrame(columns = col_names)
    
for focal_loc in locations:
    
    try:
        PopSize = StatePops[StatePops['Province/State'] == focal_loc]['PopSize'].tolist()    
        PopSize = PopSize[0]
        ArrivalDate = StatePops[StatePops['Province/State'] == focal_loc]['Date_of_first_reported_infection'].tolist()
        ArrivalDate = ArrivalDate[0]

 
    except:
        continue
        
    for model in models:
            
        new_cases = []
                    
        # A function to generate all figures and tables
                    
        # variables:
            # obs_x: observed x values
            # obs_y: observed y values
            # model: the model to fit
            # T0: likely date of first infection
            # N: population size of interest
            # ArrivalDate: likely data of first infection (used by SEIR-SD model)
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
                    
            # declare the following as global variables so their changes can be 
            # seen/used by outside functions
                    
        # filter main dataframe to include only the chosen location
        df_sub = cases_df[cases_df['Province/State'] == focal_loc]
                    
        # get column labels, will filter below to extract dates
        yi = list(df_sub)
                    
        obs_y_trunc = []
        obs_y = df_sub.iloc[0,4:].values
        
        i = 0
        while obs_y[i] == 0:
            i+=1

        obs_y = obs_y[i:]
        
        
        for i, j in enumerate(list(range(-200, 0))):
            
            try:
                        
                if j == 0:
                    # get dates for today's predictions/forecast
                    DATES = yi[4:]
                    obs_y_trunc = df_sub.iloc[0,4:].values
                
                else:
                    # get dates for previous days predictions/forecast
                    DATES = yi[4:j]
                    obs_y_trunc = df_sub.iloc[0,4:j].values
                    
                        
                # remove leading zeros from observed y values 
                # and coordinate it with dates
                ii = 0
                while obs_y_trunc[ii] == 0: 
                    ii+=1
                    
                y = obs_y_trunc[ii:]
                dates = DATES[ii:]
                
                latest_date = pd.to_datetime(dates[-1])
                first_date = pd.to_datetime(dates[0])
                
                # declare x as a list of integers from 0 to len(y)
                x = list(range(len(y)))
                
                # Call function to use chosen model to obtain:
                #    r-square for observed vs. predicted
                #    predicted y-values
                #    forecasted x and y values
                iterations = 200000
                ForecastDays = 0
                obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y, params = fxns.fit_curve(x, y, 
                                    model, ForecastDays, PopSize, ArrivalDate, j, iterations)
                            
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
            
                # get dates from ArrivalDate to the current day
                dates = pd.date_range(start=first_date, end=latest_date)
                dates = dates.strftime('%m/%d')
                    
                output_list = [dates, obs_pred_r2, model, focal_loc]
                    
                model_fits_df.loc[len(model_fits_df)] = output_list
                    
                print(focal_loc, ' : ', model, ' : ', dates[-1], ' : ', obs_pred_r2)
                
            except:
                continue
                
        print('\n')



model_fits_df.to_pickle('data/model_results_dataframe.pkl')