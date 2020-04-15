import ipywidgets as widgets # provides interactive functionality
import pandas as pd # data frame library

import datetime # library for date-time functionality
import numpy as np # numerical python
import model_fxns as fxns

import sys

#### Define the class App_GetFits
#### Will contain all other functions for modeling, calculation, and plotting

class App_GetFits:
    
    # Dataframe containing data aggregated from Johns Hopkins daily reports
    
    def __init__(self, ap_df, statepops, seir_fits_df):
        
        # df is a copy of the primary dataframe ap_df
        
        self._seir_fits_df = seir_fits_df
        self._df = ap_df
        self._statepops = statepops
        
        self.new_cases = []
        
        # locations the user can choose
        available_indicators2 = list(set(self._df['Province/State']))
        # order the locations alphabetically
        available_indicators2.sort()
        
        # Find the index of your default location
        lab = 'Illinois'
        ill = available_indicators2.index(lab)
        
        # declare widgets: dropdowns, floattexts, toggle buttons, datepicker, etc.
        self._1_dropdown = self._create_dropdown(['logistic', 'SEIR-SD', 'exponential', 'polynomial'],
                                                 0, label = 'Choose a model to fit:')
        
        self._2_dropdown = self._create_dropdown(available_indicators2, ill, label = 'Choose a location:')
        
        
        # define containers to hold the widgets, plots, and additional outputs
        self._plot_container = widgets.Output()
        
        _app_container = widgets.VBox([widgets.HBox([self._1_dropdown, self._2_dropdown], 
                             layout=widgets.Layout(align_items='flex-start', flex='0 auto auto', width='100%'))],
                           
                           
                           layout=widgets.Layout(display='flex', flex_flow='column', border='solid 2px', 
                                        align_items='stretch', width='100%'))
                           
                           
                
        # 'flex-start', 'flex-end', 'center', 'baseline', 'stretch', 'inherit', 'initial', 'unset'
        self.container = widgets.VBox([
            widgets.HBox([
                _app_container
            ])
        ], layout=widgets.Layout(align_items='flex-start', width='100%', flex='auto'))
        self._update_app()
        
        
    @classmethod
    def fits(cls, ap_df, statepops, seir_fits_df):  
        
        # reuse primary dataframe when updating the app
        return cls(ap_df, statepops, seir_fits_df)
        
        
    def _create_dropdown(self, indicators, initial_index, label):
        # create a dropdown widget
        dropdown = widgets.Dropdown(options=indicators, 
                                    layout={'width': '60%'},
                                    style={'description_width': '49%'},
                                    value=indicators[initial_index],
                                   description=label)
        
        dropdown.observe(self._on_change, names=['value'])
        return dropdown
    
    def _create_floattext(self, label, val, minv, maxv, boxw, desw):
        # create a floattext widget
        obj = widgets.BoundedFloatText(
                    value=val,
                    min=minv,
                    max=maxv,
                    description=label,
                    disabled=False,
                    layout={'width': boxw},
                    style={'description_width': desw},
                )
        obj.observe(self._on_change, names=['value'])
        return obj
    
    
    
    
    def _on_change(self, _):
        # do the following when app inputs change
        self._update_app()

    def _update_app(self):
        col_names =  ['obs_y', 'pred_y', 'forecasted_y', 'pred_dates', 'forecast_dates', 'label', 'obs_pred_r2', 'model', 
                      'focal_loc', 'PopSize', 'ArrivalDate', 'pred_clr', 'fore_clr']
        
        model_fits_df  = pd.DataFrame(columns = col_names)

        self._model_fits_df = model_fits_df
        self.new_cases = []
        self.ForecastDays = 60
        # update the app when called
        
        # redefine input/parameter values
        self.model = self._1_dropdown.value
        self.focal_loc = self._2_dropdown.value
        StatePops = self._statepops
        
        
        # wait to clear the plots/tables until new ones are generated
        self._plot_container.clear_output(wait=True)
        
        with self._plot_container:
            # Run the functions to generate figures and tables
            self._get_fit(self.focal_loc, self.ForecastDays, StatePops, self.model, self._seir_fits_df)
            
            
            
    def _get_fit(self, focal_loc, ForecastDays, StatePops, model, seir_fits_df):
        
        
        PopSize = StatePops[StatePops['Province/State'] == focal_loc]['PopSize'].tolist()
        PopSize = PopSize[0]
        
        ArrivalDate = StatePops[StatePops['Province/State'] == focal_loc]['Date_of_first_reported_infection'].tolist()
        ArrivalDate = ArrivalDate[0]
        
        SEIR_Fit = seir_fits_df[seir_fits_df['focal_loc'] == focal_loc]
        
        # A function to generate all figures and tables
        
        # variables:
            # obs_x: observed x values
            # obs_y: observed y values
            # model: the model to fit
            # T0: likely date of first infection
            # ForecastDays: number of days ahead to extend predictions
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
        
        
        # add 1 to number of forecast days for indexing purposes
        ForecastDays = int(ForecastDays+1)
        
        
        # filter main dataframe to include only the chosen location
        df_sub = self._df[self._df['Province/State'] == focal_loc]
        
        # get column labels, will filter below to extract dates
        yi = list(df_sub)
        
        
        
        
        obs_y_trunc = []
        fore_clrs =  ['darkorchid', 'blue', 'green', 'orange', 'red']
        pred_clrs = ['0.1', '0.2', '0.4', '0.6', '0.8']
        
        for i, j in enumerate([-4,-3,-2,-1, 0]):
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
            
            
            
            # remove leading zeros from observed y values 
            # and coordinate it with dates
            #y = []
            #dates = []
            #for ii, val in enumerate(obs_y_trunc):
            #    if len(y) > 0 or val > 0:
            #        y.append(val)
            #       dates.append(DATES[ii])
            
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
            iterations = 1000
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
                            self.new_cases.append(forecasted_y[i] - forecasted_y[i-1])
                        else:
                            self.new_cases.append(0)
                    if i == 0:
                        self.new_cases.append(forecasted_y[i])
                        
                
            # get dates from ArrivalDate to the current day
            dates = pd.date_range(start=first_date, end=latest_date)
            dates = dates.strftime('%m/%d')
            
            
            output_list = [y, pred_y, forecasted_y, dates, fdates,
                           label, obs_pred_r2, model, focal_loc, PopSize, 
                           ArrivalDate, pred_clr, fore_clr]
            
            self._model_fits_df.loc[len(self._model_fits_df)] = output_list
            
        
        