import ipywidgets as widgets # provides interactive functionality
from IPython.display import HTML, display, clear_output # display and html functionality
import matplotlib.pyplot as plt # plotting library
import pandas as pd # data frame library

import csv # csv import functionality
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
import sys # used to exit the program; for testing 

import model_fxns as fxns



#### Define the class App_GetFits
#### Will contain all other functions for modeling, calculation, and plotting

class App_GetFits:
    
    
    
    # Dataframe containing data aggregated from Johns Hopkins daily reports
    
    
    #### delcare objects intended to be used as global variables 

    
    # declare the following as global so they can be shared between functions
    # and classes
    
    
    def __init__(self, df):
        
        self.itsProblem = "problem"
        
        self.Forecasted_cases_df_for_download = []
        self.Forecasted_patient_census_df_for_download = []
        self.Forecasted_ppe_needs_df_for_download = []
        self.model = str()
        
        # df is a copy of the primary dataframe ap_df
        self._df = df
        
        # locations the user can choose
        available_indicators2 = list(set(self._df['Province/State']))
        # order the locations alphabetically
        available_indicators2.sort()
        
        # Find the index of your default location
        ill = available_indicators2.index('Illinois')
        
        # declare widgets: dropdowns, floattexts, toggle buttons, datepicker, etc.
        self._1_dropdown = self._create_dropdown(['logistic', 'SEIR-SD', 'exponential', 'polynomial'],
                                                 0, label = 'Choose a model to fit:')
        self._2_dropdown = self._create_dropdown(available_indicators2, ill, label = 'Choose a location:')
        self._3_floattext = self._create_floattext(label = 'p% Visiting your hospital:', 
                                                   val=10, minv=0, maxv=100, boxw='33%', desw='70%')
        self._4_floattext = self._create_floattext(label = '% Admitted to your hospital:', 
                                                   val=30, minv=0, maxv=100, boxw='33%', desw='70%')
        self._5_floattext = self._create_floattext(label = '% Admitted to critical care:', 
                                                   val=25, minv=0, maxv=100, boxw='33%', desw='70%')
        self._6_floattext = self._create_floattext(label = 'LOS (non-critical care):', 
                                                   val=3, minv=1, maxv=180, boxw='33%', desw='70%')
        self._7_floattext = self._create_floattext(label = 'LOS (critical care):', 
                                                   val=12, minv=1, maxv=180, boxw='33%', desw='70%')
        self._8_floattext = self._create_floattext(label = '% of ICU on vent:',
                                                   val=60, minv=0, maxv=100, boxw='33%', desw='70%')
        self._9_toggle = self._create_toggle()
        
        self._10_floattext = self._create_floattext(label = 'GLOVE SURGICAL', 
                                                    val=2, minv=0, maxv=1000, boxw='33%', desw='70%')
        self._11_floattext = self._create_floattext(label = 'GLOVE EXAM NITRILE', 
                                                    val=260, minv=0, maxv=1000, boxw='33%', desw='70%')
        self._12_floattext = self._create_floattext(label = 'GLOVE EXAM VINYL', 
                                                    val=10, minv=0, maxv=1000, boxw='33%', desw='70%')
        self._13_floattext = self._create_floattext(label = 'MASK FACE PROC. ANTI FOG', 
                                                    val=45, minv=0, maxv=1000, boxw='33%', desw='70%')
        self._14_floattext = self._create_floattext(label = 'MASK PROC. FLUID RESISTANT', 
                                                    val=1, minv=0, maxv=1000, boxw='33%', desw='70%')
        self._15_floattext = self._create_floattext(label = 'GOWN ISOLATION XL YELLOW', 
                                                    val=2, minv=0, maxv=1000, boxw='33%', desw='70%')
        self._16_floattext = self._create_floattext(label = 'MASK SURG. ANTI FOG W/FILM', 
                                                    val=1, minv=0, maxv=1000, boxw='33%', desw='70%')
        self._17_floattext = self._create_floattext(label = 'SHIELD FACE FULL ANTI FOG', 
                                                    val=1, minv=0, maxv=1000, boxw='33%', desw='70%')
        self._18_floattext = self._create_floattext(label = 'RESP. PART. FILTER REG', 
                                                    val=11, minv=0, maxv=1000, boxw='33%', desw='70%')
        
        self._19_floattext = self._create_floattext(label = 'Forecast length (days)', 
                                                    val=10, minv=1, maxv=30, boxw='33%', desw='70%')
        
        self._20_floattext = self._create_floattext(label = 'Focal population size', 
                                                    val=12740000, minv=1, maxv=10**8, boxw='50%', desw='50%')
        self._22_floattext = self._create_floattext(label = 'Incubation period', 
                                                    val=5, minv=1, maxv=21, boxw='50%', desw='50%')
        self._23_floattext = self._create_floattext(label = 'Infectious period', 
                                                    val=7, minv=1, maxv=21, boxw='50%', desw='50%')
        self._24_floattext = self._create_floattext(label = 'Reproduction no.', 
                                                    val=4, minv=1, maxv=21, boxw='50%', desw='50%')
        self._25_floattext = self._create_floattext(label = 'Social Distancing: 1 - 100', 
                                                    val=10, minv=0, maxv=100, boxw='50%', desw='50%')
        self._26_floattext = self._create_floattext(label = 'Avg. visit time lag (days)', 
                                                    val=0, minv=0, maxv=14, boxw='33%', desw='70%')
        
        self._21_datepickr = self._create_datepickr(label='Likely date of 1st exposure', boxw='49%', desw='51%')
        
        # define containers to hold the widgets, plots, and additional outputs
        self._plot_container = widgets.Output()
        
        _app_container = widgets.VBox(
            [widgets.VBox([widgets.HBox([self._9_toggle, self._1_dropdown, self._2_dropdown], 
                             layout=widgets.Layout(align_items='flex-start', flex='0 auto auto', width='100%')),
                           
                           widgets.HBox([self._3_floattext, self._4_floattext, self._5_floattext],
                             layout=widgets.Layout(align_items='flex-start', flex='0 0 auto', width='100%')),
                           
                           widgets.HBox([self._6_floattext, self._7_floattext, self._8_floattext],
                             layout=widgets.Layout(align_items='flex-start', flex='0 0 auto', width='100%')),
                          
                           widgets.HBox([self._10_floattext, self._11_floattext, self._12_floattext],
                             layout=widgets.Layout(align_items='flex-start', flex='0 0 auto', width='100%')),
                           
                           widgets.HBox([self._13_floattext, self._14_floattext, self._15_floattext],
                             layout=widgets.Layout(align_items='flex-start', flex='0 0 auto', width='100%')),
                           
                           widgets.HBox([self._16_floattext, self._17_floattext, self._18_floattext],
                             layout=widgets.Layout(align_items='flex-start', flex='0 0 auto', width='100%')),
                          
                           widgets.HBox([self._19_floattext, self._26_floattext],
                             layout=widgets.Layout(align_items='flex-start', flex='0 0 auto', width='100%'))],
                           
                           layout=widgets.Layout(display='flex', flex_flow='column', border='solid 1px', 
                                        align_items='stretch', width='100%')),
             
             widgets.VBox([widgets.HBox([self._20_floattext, self._22_floattext],
                             layout=widgets.Layout(align_items='flex-start', flex='0 0 auto', width='100%')),
                           widgets.HBox([self._23_floattext, self._24_floattext],
                             layout=widgets.Layout(align_items='flex-start', flex='0 0 auto', width='100%')),
                           widgets.HBox([self._25_floattext, self._21_datepickr],
                             layout=widgets.Layout(align_items='flex-start', flex='0 0 auto', width='100%'))],
                           
                           layout=widgets.Layout(display='flex', flex_flow='column', border='solid 1px', 
                                        align_items='stretch', width='100%')),
                           self._plot_container], layout=widgets.Layout(display='flex', flex_flow='column', 
                                        border='solid 2px', align_items='initial', width='100%'))
                
        # 'flex-start', 'flex-end', 'center', 'baseline', 'stretch', 'inherit', 'initial', 'unset'
        self.container = widgets.VBox([
            widgets.HBox([
                _app_container
            ])
        ], layout=widgets.Layout(align_items='flex-start', flex='0 0 auto', width='100%'))
        self._update_app()
        
        
    @classmethod
    def from_url(cls):
        ap_df = pd.read_csv('COVID-CASES-DF.txt', sep='\t') 
        ap_df = ap_df[ap_df['Country/Region'] == 'US']
        ap_df = ap_df[ap_df['Province/State'] != 'US']
        ap_df.drop(columns=['Unnamed: 0'], inplace=True)
        # reuse primary dataframe when updating the app
        return cls(ap_df)
        
        
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
    
    
    
    def _create_toggle(self): 
        # create a toggle button widget
        obj = widgets.ToggleButton(
                    value=False,
                    description='log-scale',
                    disabled=False,
                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
                    tooltip='Description',
                    icon='check' # (FontAwesome names without the `fa-` prefix)
                )
        obj.observe(self._on_change, names=['value'])
        return obj
    
    
    def _create_datepickr(self, label, boxw, desw):
        # create a widget for picking dates
        obj = widgets.DatePicker(
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
        # update the app when called
        
        # redefine input/parameter values
        self.model = self._1_dropdown.value
        focal_loc = self._2_dropdown.value
        per_loc  = self._3_floattext.value
        per_admit = self._4_floattext.value
        per_cc = self._5_floattext.value
        LOS_nc = self._6_floattext.value
        LOS_cc = self._7_floattext.value
        per_vent = self._8_floattext.value
        log_scl = self._9_toggle.value
        
        ppe_GLOVE_SURGICAL = self._10_floattext.value
        ppe_GLOVE_EXAM_NITRILE = self._11_floattext.value
        ppe_GLOVE_GLOVE_EXAM_VINYL = self._12_floattext.value
        ppe_MASK_FACE_PROCEDURE_ANTI_FOG= self._13_floattext.value
        ppe_MASK_PROCEDURE_FLUID_RESISTANT = self._14_floattext.value
        ppe_GOWN_ISOLATION_XLARGE_YELLOW= self._15_floattext.value
        ppe_MASK_SURGICAL_ANTI_FOG_W_FILM = self._16_floattext.value
        ppe_SHIELD_FACE_FULL_ANTI_FOG = self._17_floattext.value
        ppe_RESPIRATOR_PARTICULATE_FILTER_REG = self._18_floattext.value
        ForecastDays = self._19_floattext.value
        State_Pop_Size = self._20_floattext.value
        T0 = self._21_datepickr.value
        Incubation = self._22_floattext.value
        Infectious = self._23_floattext.value
        Rho = self._24_floattext.value
        SocialDist = self._25_floattext.value
        TimeLag = self._26_floattext.value
        
        # wait to clear the plots/tables until new ones are generated
        self._plot_container.clear_output(wait=True)
        
        with self._plot_container:
            # Run the functions to generate figures and tables
            self._get_fit(focal_loc, per_loc, per_admit, per_cc, LOS_cc, LOS_nc, per_vent, log_scl,
                         ppe_GLOVE_SURGICAL, ppe_GLOVE_EXAM_NITRILE, ppe_GLOVE_GLOVE_EXAM_VINYL,
                         ppe_MASK_FACE_PROCEDURE_ANTI_FOG, ppe_MASK_PROCEDURE_FLUID_RESISTANT, 
                         ppe_GOWN_ISOLATION_XLARGE_YELLOW, ppe_MASK_SURGICAL_ANTI_FOG_W_FILM,
                         ppe_SHIELD_FACE_FULL_ANTI_FOG, ppe_RESPIRATOR_PARTICULATE_FILTER_REG,
                         ForecastDays, State_Pop_Size, T0, Incubation, Infectious, Rho, SocialDist,
                         TimeLag)
            
            plt.show()
            
            
    def _get_fit(self, focal_loc, per_loc, per_admit, per_cc, LOS_cc, LOS_nc, per_vent, log_scl,
                        ppe_GLOVE_SURGICAL, ppe_GLOVE_EXAM_NITRILE, ppe_GLOVE_GLOVE_EXAM_VINYL,
                        ppe_MASK_FACE_PROCEDURE_ANTI_FOG, ppe_MASK_PROCEDURE_FLUID_RESISTANT, 
                        ppe_GOWN_ISOLATION_XLARGE_YELLOW, ppe_MASK_SURGICAL_ANTI_FOG_W_FILM,
                        ppe_SHIELD_FACE_FULL_ANTI_FOG, ppe_RESPIRATOR_PARTICULATE_FILTER_REG,
                        ForecastDays, State_Pop_Size, T0, Incubation, Infectious, Rho, SocialDist,
                        TimeLag):
        
        
        # A function to generate all figures and tables
        
        # variables:
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
        
        # declare the following as global variables so their changes can be 
        # seen/used by outside functions
        
        
        # add 1 to number of forecast days for indexing purposes
        ForecastDays = int(ForecastDays+1)
        
        # declare figure object
        fig = plt.figure(figsize=(11, 17))
        # use subplot2grid functionality
        ax = plt.subplot2grid((6, 4), (0, 0), colspan=2, rowspan=2)
        
        # filter main dataframe to include only the chosen location
        df_sub = self._df[self._df['Province/State'] == focal_loc]
        # get column labels, will filter below to extract dates
        yi = list(df_sub)
        
        # declare colors for plotting predictions and forecasts in figure 1 
        clrs =  ['mistyrose', 'pink', 'lightcoral', 'salmon', 'red']
        clrs2 = ['powderblue', 'lightskyblue', 'cornflowerblue', 'dodgerblue', 'blue']
        
        # Generate previous days predictions and forecasts
        # 0 is the current day, -1 is yesterday, etc.
        obs_y_trunc = []
        for i, j in enumerate([-4,-3,-2,-1, 0]):
            if j == 0:
                # get dates for today's predictions/forecast
                DATES = yi[4:]
                obs_y_trunc = df_sub.iloc[0,4:].values
            else:
                # get dates for previous days predictions/forecast
                DATES = yi[4:j]
                obs_y_trunc = df_sub.iloc[0,4:j].values
            
            if self.model == 'SEIR-SD':
                # prediction and forecasts from the SEIR-SD
                # model do not change over time
                DATES = yi[4:]
                obs_y_trunc = df_sub.iloc[0,4:].values
            
            # remove leading zeros from observed y values 
            # and coordinate it with dates
            y = []
            dates = []
            for ii, val in enumerate(obs_y_trunc):
                if len(y) > 0 or val > 0:
                    y.append(val)
                    dates.append(DATES[ii])
            
            # declare x as a list of integers from 0 to len(y)
            x = list(range(len(y)))

            # Call function to use chosen model to obtain:
            #    r-square for observed vs. predicted
            #    predicted y-values
            #    forecasted x and y values
            obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y = fxns.fit_curve(x, y, 
                                self.model, ForecastDays, State_Pop_Size, T0, Incubation, Infectious, Rho, SocialDist)
            
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
            forecast_vals = np.copy(forecasted_y)

            # number of from T0 to end of forecast window
            numdays = len(forecasted_x)
            latest_date = pd.to_datetime(dates[-1])
            first_date = pd.to_datetime(dates[0])

            # get the date of the last day in the forecast window
            future_date = latest_date + datetime.timedelta(days = ForecastDays-1)
            
            # get all dates from T0 to the last day in the forecast window
            fdates = pd.date_range(start=first_date, end=future_date)
            fdates = fdates.strftime('%m/%d')
            
            # designature plot label for legend
            if j == 0:
                label='Current forecast'
                if self.model == 'SEIR-SD':
                    label = 'Forecast\n'
            elif self.model != 'SEIR-SD':
                label = str(-j)+' day old forecast'
            elif self.model == 'SEIR-SD':
                label = None
            
            # plot forecasted y values vs. dates
            plt.plot(fdates, forecasted_y, c=clrs[i], linewidth=3, label=label)
            
            # get dates from T0 to the current day
            dates = pd.date_range(start=first_date, end=latest_date)
            dates = dates.strftime('%m/%d')
            
            # plot predicted y values vs. dates
            plt.plot(dates, pred_y, c=clrs2[i], linewidth=3)
            plt.scatter(dates, y, c='0.2', s=100, alpha=0.8, linewidths=0.1)
            
        
        # For new cases, subtract today's number from yesterday's
        # in the case of negative values, reassign as 0
        new_cases = []
        for i, val in enumerate(forecast_vals):
            if i > 0:
                if forecast_vals[i] - forecast_vals[i-1] > 0:
                    new_cases.append(forecast_vals[i] - forecast_vals[i-1])
                else:
                    new_cases.append(0)
            if i == 0:
                new_cases.append(forecast_vals[i])

        # declare figure legend
        leg = ax.legend(handlelength=0, handletextpad=0, fancybox=False,
                        loc=2, frameon=False, fontsize=12)

        # color legend text by the color of the line
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())

        # set line from legend handle as invisible
        for item in leg.legendHandles: 
            item.set_visible(False)

        plt.xticks(rotation=35, ha='center')
        plt.xlabel('Date', fontsize=14, fontweight='bold')
        plt.ylabel('Confirmed cases', fontsize=14, fontweight='bold')
        
        # log-scale y-values to base 10 if user choose the option
        if log_scl == True:
            plt.yscale('log')

        # modify number of dates displayed on x-axis
        # to avoid over-crowding the axis
        if len(forecasted_x) < 10:
            i = 1
        elif len(forecasted_x) < 20:
            i = 4
        elif len(forecasted_x) < 40:
            i = 6
        else:
            i = 8

        ax = plt.gca()
        temp = ax.xaxis.get_ticklabels()
        temp = list(set(temp) - set(temp[::i]))
        for label in temp:
            label.set_visible(False)
        
        # cutomize plot title
        if self.model == 'SEIR-SD':
            label = 'Model fitting, current ' + r'$r^{2}$' + ' = ' + str(np.round(obs_pred_r2, 2))
            label += '\n(model under development)'
            plt.title(label, fontsize = 14, fontweight = 'bold')
        else:
            plt.title('Model fitting, current ' + r'$r^{2}$' + ' = ' + str(np.round(obs_pred_r2, 2)), fontsize = 16, fontweight = 'bold')
        
        
        
        # Declare figure axis to hold table of forecasted cases, visits, admits
        ax = plt.subplot2grid((6, 4), (0, 2), colspan=2, rowspan=2)
        # The figure will actually be a table so turn the figure axes off
        ax.axis('off')

        # shorten location name if longer than 12 characters
        loc = str(focal_loc)
        if len(loc) > 12:
            loc = loc[:12]
            loc = loc + '...'

        # declare column labels
        col_labels = ['Total cases', 'New cases', 'New visits', 'New admits']

        # row labels are the dates
        row_labels = fdates.tolist()  
        # only show the current date and dates in the forecast window
        row_labels = row_labels[-(ForecastDays):]
        # truncate forecasted_y to only the current day and days 
        # in the forecast window
        sub_f = forecasted_y[-(ForecastDays):]
        
        # lists to hold table values
        table_vals = []
        cclr_vals = []
        rclr_vals = []
        
        #### Inclusion of time lag
        # time lag is modeled as a Poisson distributed 
        # random variable with a mean chosen by the user (TimeLag)
        new_cases_lag = []
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
        
        # lower truncate lists for forecast window
        # that is, do not include days before present day
        new_cases = new_cases[-(ForecastDays):]
        forecasted_y = forecasted_y[-(ForecastDays):]
        ts_lag2 = ts_lag[-(ForecastDays):]
        
        # Declare pandas dataframe to hold data for download
        self.Forecasted_cases_df_for_download = pd.DataFrame(columns = ['date'] + col_labels)
        
        # For each date intended for the output table
        for i in range(len(row_labels)):
            
            new = new_cases[i]
            val = ts_lag2[i]
            
            # each cell is a row with 4 columns:
            #     Total cases, 
            #     new cases, 
            #     time-lagged visits to your hospital,
            #     time-lagged admits to your hospital
            
            cell = [int(np.round(sub_f[i])), 
                    int(np.round(new)), 
                    int(np.round(val * (per_loc * 0.01))),
                    int(np.round((0.01 * per_admit) * val * (per_loc * 0.01)))]
            
            # Add the row to the dataframe
            df_row = [row_labels[i]]
            df_row.extend(cell)
            labs = ['date'] + col_labels
            temp = pd.DataFrame([df_row], columns=labs)
            self.Forecasted_cases_df_for_download = pd.concat([self.Forecasted_cases_df_for_download, temp])
            
            # color the first row grey and remaining rows white
            if i == 0:
                rclr = '0.8'
                cclr = ['0.8', '0.8', '0.8', '0.8']
            else:
                rclr = 'w'
                cclr = ['w', 'w', 'w', 'w']
            table_vals.append(cell)
            cclr_vals.append(cclr)
            rclr_vals.append(rclr)

        # Generate and customize table for output
        ncol = 4
        lim = ForecastDays
        if lim > 18:
            lim = 19
            
        the_table = plt.table(cellText=table_vals[0:lim],
                        colWidths=[0.32, 0.32, 0.32, 0.32],
                        rowLabels=row_labels[0:lim],
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='upper center',
                        cellColours=cclr_vals[0:lim],
                        rowColours =rclr_vals[0:lim])
        
        the_table.auto_set_font_size(True)
        the_table.scale(1, 1.32)
        
        # Customize table title
        if ForecastDays <= 18:
            plt.title('Forecasted cases for '+ loc, fontsize = 16, fontweight = 'bold')
        elif ForecastDays > 18:
            titletext = 'Forecasted cases for '+ loc + '\ndata beyond 18 days is available in the csv (below)'
            plt.title(titletext, fontsize = 14, fontweight = 'bold')
            
        
        
        
        # Generate figure for patient census
        ax = plt.subplot2grid((6, 4), (2, 0), colspan=2, rowspan=2)
        
        #### Construct arrays for critical care and non-critical care patients
        cc = (0.01 * per_cc) * (0.01 * per_admit) * (0.01 * per_loc) * np.array(ts_lag)
        cc = cc.tolist()
        
        nc = (1 - (0.01 * per_cc)) * (0.01 * per_admit) * (0.01 * per_loc) * np.array(ts_lag)
        nc = nc.tolist()
        
        # LOS for non critical care = 5 days
        # LOS for critical care = 10 days
        
        
        
        # Model length of stay (LOS) as a binomially distributed
        # random variable according to binomial parameters p and n
        #    p: used to obtain a symmetrical distribution 
        #    n: (n_cc & n_nc) = 2 * LOS will produce a binomial
        #       distribution with a mean equal to the LOS
        
        p = 0.5
        n_cc = LOS_cc*2
        n_nc = LOS_nc*2
        
        # get the binomial random variable properties
        rv_nc = binom(n_nc, p)
        # Use the binomial cumulative distribution function
        p_nc = rv_nc.cdf(np.array(range(1, len(fdates)+1)))
        
        # get the binomial random variable properties
        rv_cc = binom(n_cc, p)
        # Use the binomial cumulative distribution function
        p_cc = rv_cc.cdf(np.array(range(1, len(fdates)+1)))
        
        # Initiate lists to hold numbers of critical care and non-critical care patients
        # who are expected as new admits (index 0), as 1 day patients, 2 day patients, etc.
        LOScc = np.zeros(len(fdates))
        LOScc[0] = ts_lag[0] * (0.01 * per_cc) * (0.01 * per_admit) * (0.01 * per_loc)
        LOSnc = np.zeros(len(fdates))
        LOSnc[0] =  ts_lag[0] * (1-(0.01 * per_cc)) * (0.01 * per_admit) * (0.01 * per_loc)
        
        total_nc = []
        total_cc = []
        
        # Roll up patient carry-over into lists of total critical care and total
        # non-critical patients expected
        for i, day in enumerate(fdates):
            LOScc = LOScc * (1 - p_cc)
            LOSnc = LOSnc * (1 - p_nc)
            
            LOScc = np.roll(LOScc, shift=1)
            LOSnc = np.roll(LOSnc, shift=1)
            
            LOScc[0] = ts_lag[i] * (0.01 * per_cc) * (0.01 * per_admit) * (0.01 * per_loc)
            LOSnc[0] = ts_lag[i] * (1 - (0.01 * per_cc)) * (0.01 * per_admit) * (0.01 * per_loc)
    
            total_nc.append(np.sum(LOSnc))
            total_cc.append(np.sum(LOScc))
            
        # Plot the critical care and non-critical care patient census over the 
        # forecasted time frame
        plt.plot(fdates, total_cc, c='Crimson', label='Critical care', linewidth=3)
        plt.plot(fdates, total_nc, c='0.3', label='Non-critical care', linewidth=3)
        plt.title('Forecasted census', fontsize = 16, fontweight = 'bold')
        
        # log-scale y-values to base 10 if the user has chosen
        if log_scl == True:
            plt.yscale('log')
        
        # As before, limit dates displayed on the x-axis
        # prevents overcrowding
        ax = plt.gca()
        temp = ax.xaxis.get_ticklabels()
        temp = list(set(temp) - set(temp[::8]))
        for label in temp:
            label.set_visible(False)
            
        # As before, remove legend line handles and change the color of 
        # the text to match the color of the line
        leg = ax.legend(handlelength=0, handletextpad=0, fancybox=False,
                        loc='best', frameon=False, fontsize=14)

        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())

        for item in leg.legendHandles: 
            item.set_visible(False)
        
        plt.ylabel('COVID-19 patients', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=14, fontweight='bold')
        
        
        
        
        # Declare axis to be used for patient census table
        # and turn the visibility off
        ax = plt.subplot2grid((6, 4), (2, 2), colspan=2, rowspan=2)
        ax.axis('off')
        
        # Truncate location names if longer than 12 characters
        if len(loc) > 12:
            loc = loc[:12]
            loc = loc + '...'

        # declare table column labels
        col_labels = ['All COVID', 'Non-ICU', 'ICU', 'Vent']

        # declare row labels as dates
        row_labels = fdates.tolist()
        
        # truncate row labels and values to the present day
        # and days in the forecast window
        row_labels = row_labels[-(ForecastDays):]
        total_nc_trunc = total_nc[-(ForecastDays):]
        total_cc_trunc = total_cc[-(ForecastDays):]
        
        # declare lists to hold table cell values and
        # row colors
        table_vals, cclr_vals, rclr_vals = [], [], []
        
        # declare pandas dataframe to hold patient census data for download
        self.Forecasted_patient_census_df_for_download = pd.DataFrame(columns = ['date'] + col_labels)
        # For each row...
        for i in range(len(row_labels)):
            # Each cell is a row that holds:
            #    Total number of admits expected,
            #    Total number of non-critical care COVID-19 patients expected
            #    Total number of critical care COVID-19 patents expected
            #    Total number of ICU patients on ventilators expected
            cell = [int(np.round(total_nc_trunc[i] + total_cc_trunc[i])), 
                    int(np.round(total_nc_trunc[i])),
                    int(np.round(total_cc_trunc[i])), 
                    int(np.round(total_cc_trunc[i]*(0.01*per_vent)))]
            
            # add the cell to the dataframe intended for csv download
            df_row = [row_labels[i]]
            df_row.extend(cell)
            labs = ['date'] + col_labels
            temp = pd.DataFrame([df_row], columns=labs)
            self.Forecasted_patient_census_df_for_download = pd.concat([self.Forecasted_patient_census_df_for_download, temp])
            
            # set colors of rows
            if i == 0:
                rclr = '0.8'
                cclr = ['0.8', '0.8', '0.8', '0.8']
            else:
                rclr = 'w'
                cclr = ['w', 'w', 'w', 'w']
                
            # append cells and colors to respective lists
            table_vals.append(cell)
            cclr_vals.append(cclr)
            rclr_vals.append(rclr)
            
        # limit the number of displayed table rows    
        ncol = 4
        lim = ForecastDays
        if lim > 18:
            lim = 19
        
        # declare and customize the table
        the_table = plt.table(cellText=table_vals[0:lim],
                        colWidths=[0.255, 0.255, 0.255, 0.255],
                        rowLabels=row_labels[0:lim],
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='upper center',
                        cellColours=cclr_vals[0:lim],
                        rowColours =rclr_vals[0:lim])
        
        the_table.auto_set_font_size(True)
        the_table.scale(1, 1.32)
        
        # Set the plot (table) title
        if ForecastDays <= 18:
            plt.title('Beds needed for COVID-19 cases', fontsize = 16, fontweight = 'bold')
        elif ForecastDays > 18:
            titletext = 'Beds needed for COVID-19 cases' + '\ndata beyond 18 days is available in the csv (below)'
            plt.title(titletext, fontsize = 14, fontweight = 'bold')
            
        
        
        
        ####################### PPE ##################################
        ax = plt.subplot2grid((6, 4), (4, 0), colspan=2, rowspan=2)
        
        #### Construct arrays for critical care and non-critical care patients
        
        # All covid patients expected in house on each forecasted day. PUI is just a name here
        PUI_COVID = np.array(total_nc) + np.array(total_cc) 
        # Preparing to add new visits, fraction of new cases visiting your hospital = 0.01 * per_loc 
        new_visits_your_hospital = ts_lag * (0.01 * per_loc)
        # Add number of new visits to number of in house patients
        PUI_COVID = PUI_COVID + new_visits_your_hospital
        
        glove_surgical = np.round(ppe_GLOVE_SURGICAL * PUI_COVID).astype('int')
        glove_nitrile = np.round(ppe_GLOVE_EXAM_NITRILE * PUI_COVID).astype('int')
        glove_vinyl = np.round(ppe_GLOVE_GLOVE_EXAM_VINYL * PUI_COVID).astype('int')
        face_mask = np.round(ppe_MASK_FACE_PROCEDURE_ANTI_FOG * PUI_COVID).astype('int')
        procedure_mask = np.round(ppe_MASK_PROCEDURE_FLUID_RESISTANT * PUI_COVID).astype('int')
        isolation_gown = np.round(ppe_GOWN_ISOLATION_XLARGE_YELLOW * PUI_COVID).astype('int')
        surgical_mask = np.round(ppe_MASK_SURGICAL_ANTI_FOG_W_FILM * PUI_COVID).astype('int')
        face_shield = np.round(ppe_SHIELD_FACE_FULL_ANTI_FOG * PUI_COVID).astype('int')
        respirator = np.round(ppe_RESPIRATOR_PARTICULATE_FILTER_REG * PUI_COVID).astype('int')
        
        
        ppe_ls =[[glove_surgical, 'GLOVE SURGICAL', 'r'],
             [glove_nitrile, 'GLOVE EXAM NITRILE', 'orange'],
             [glove_vinyl, 'GLOVE EXAM VINYL', 'goldenrod'],
             [face_mask, 'MASK FACE PROCEDURE ANTI FOG', 'limegreen'],
             [procedure_mask, 'MASK PROCEDURE FLUID RESISTANT', 'green'],
             [isolation_gown, 'GOWN ISOLATION XLARGE YELLOW', 'cornflowerblue'],
             [surgical_mask, 'MASK SURGICAL ANTI FOG W/FILM', 'blue'],
             [face_shield, 'SHIELD FACE FULL ANTI FOG', 'plum'],
             [respirator, 'RESPIRATOR PARTICULATE FILTER REG', 'darkviolet']]
        
        linestyles = ['dashed', 'dotted', 'dashdot', 
                      'dashed', 'dotted', 'dashdot',
                      'dotted', 'dashed', 'dashdot']
        
        for i, ppe in enumerate(ppe_ls):
            plt.plot(fdates, ppe[0], c=ppe[2], label=ppe[1], linewidth=2, ls=linestyles[i])
    
        plt.title('Forecasted PPE needs', fontsize = 16, fontweight = 'bold')
        #if log_scl == True:
        #    plt.yscale('log')
        
        for label in ax.xaxis.get_ticklabels()[::8]:
            label.set_visible(False)

        ax = plt.gca()
        temp = ax.xaxis.get_ticklabels()
        temp = list(set(temp) - set(temp[::8]))
        for label in temp:
            label.set_visible(False)
            
        leg = ax.legend(handlelength=0, handletextpad=0, fancybox=True,
                        loc=2, frameon=True, fontsize=8)

        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())

        for item in leg.legendHandles: 
            item.set_visible(False)
        
        plt.ylabel('PPE Supplies', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=14, fontweight='bold')
        if log_scl == True:
            plt.yscale('log')
        
        
        
        
        
        ax = plt.subplot2grid((6, 4), (4, 2), colspan=2, rowspan=2)
        ax.axis('off')
        #ax.axis('tight')
        
        #### Construct arrays for critical care and non-critical care patients
        #PUI_COVID = np.array(total_nc) + np.array(total_cc)
        PUI_COVID = PUI_COVID[-(ForecastDays):]
        
        glove_surgical = np.round(ppe_GLOVE_SURGICAL * PUI_COVID).astype('int')
        glove_nitrile = np.round(ppe_GLOVE_EXAM_NITRILE * PUI_COVID).astype('int')
        glove_vinyl = np.round(ppe_GLOVE_GLOVE_EXAM_VINYL * PUI_COVID).astype('int')
        face_mask = np.round(ppe_MASK_FACE_PROCEDURE_ANTI_FOG * PUI_COVID).astype('int')
        procedure_mask = np.round(ppe_MASK_PROCEDURE_FLUID_RESISTANT * PUI_COVID).astype('int')
        isolation_gown = np.round(ppe_GOWN_ISOLATION_XLARGE_YELLOW * PUI_COVID).astype('int')
        surgical_mask = np.round(ppe_MASK_SURGICAL_ANTI_FOG_W_FILM * PUI_COVID).astype('int')
        face_shield = np.round(ppe_SHIELD_FACE_FULL_ANTI_FOG * PUI_COVID).astype('int')
        respirator = np.round(ppe_RESPIRATOR_PARTICULATE_FILTER_REG * PUI_COVID).astype('int')
        
        
        ppe_ls =[[glove_surgical, 'GLOVE SURGICAL', 'r'],
             [glove_nitrile, 'GLOVE EXAM NITRILE', 'orange'],
             [glove_vinyl, 'GLOVE EXAM VINYL', 'goldenrod'],
             [face_mask, 'MASK FACE PROCEDURE ANTI FOG', 'limegreen'],
             [procedure_mask, 'MASK PROCEDURE FLUID RESISTANT', 'green'],
             [isolation_gown, 'GOWN ISOLATION XLARGE YELLOW', 'cornflowerblue'],
             [surgical_mask, 'MASK SURGICAL ANTI FOG W/FILM', 'blue'],
             [face_shield, 'SHIELD FACE FULL ANTI FOG', 'plum'],
             [respirator, 'RESPIRATOR PARTICULATE FILTER REG', 'darkviolet']]
        
        
        if len(loc) > 12:
            loc = loc[:12]
            loc = loc + '...'

        col_labels = [ppe_ls[0][1], ppe_ls[1][1], ppe_ls[2][1], 
                      ppe_ls[3][1], ppe_ls[4][1], ppe_ls[5][1],
                      ppe_ls[6][1], ppe_ls[7][1], ppe_ls[8][1]]

        row_labels = fdates.tolist()        
        row_labels = row_labels[-(ForecastDays):]
        
        table_vals = []
        cclr_vals = []
        rclr_vals = []
        
        self.Forecasted_ppe_needs_df_for_download = pd.DataFrame(columns = ['date'] + col_labels)
        for i in range(len(row_labels)):
                
            cell = [ppe_ls[0][0][i], ppe_ls[1][0][i], ppe_ls[2][0][i], 
                      ppe_ls[3][0][i], ppe_ls[4][0][i], ppe_ls[5][0][i],
                      ppe_ls[6][0][i], ppe_ls[7][0][i], ppe_ls[8][0][i]]
            
            df_row = [row_labels[i]]
            df_row.extend(cell)
            
            labs = ['date'] + col_labels
            temp = pd.DataFrame([df_row], columns=labs)
            self.Forecasted_ppe_needs_df_for_download = pd.concat([self.Forecasted_ppe_needs_df_for_download, temp])
            
            if i == 0:
                rclr = '0.8'
                cclr = ['0.8', '0.8', '0.8', '0.8', '0.8', '0.8', '0.8', '0.8', '0.8']
            else:
                rclr = 'w'
                cclr = ['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w']
                
            table_vals.append(cell)
            cclr_vals.append(cclr)
            rclr_vals.append(rclr)
            
        ncol = 9
        cwp = 0.15
        lim = ForecastDays
        if lim > 18:
            lim = 19
            
        the_table = plt.table(cellText=table_vals[0:lim],
                        colWidths=[cwp]*9,
                        rowLabels=row_labels[0:lim],
                        colLabels=None,
                        cellLoc='center',
                        loc='upper center',
                        cellColours=cclr_vals[0:lim],
                        rowColours =rclr_vals[0:lim])
        
        the_table.auto_set_font_size(True)
        the_table.scale(1, 1.32)
        
        for i in range(len(ppe_ls)):
            clr = ppe_ls[i][2]
            for j in range(lim):
                the_table[(j, i)].get_text().set_color(clr)
        
        # set values for diagonal column labels
        hoffset = -0.4 #find this number from trial and error
        voffset = 1.0 #find this number from trial and error
        col_width = [0.06, 0.09, 0.09, 0.12, 0.133, 0.138, 0.128, 0.135, 0.142]
        
        col_labels2 =[['GLOVE SURGICAL', 'r'],
             ['GLOVE EXAM NITRILE', 'orange'],
             ['GLOVE GLOVE EXAM VINYL', 'goldenrod'],
             ['MASK FACE PROC. A-FOG', 'limegreen'],
             ['MASK PROC. FLUID RES.', 'green'],
             ['GOWN ISO. XL YELLOW', 'cornflowerblue'],
             ['MASK SURG. ANTI FOG W/FILM', 'blue'],
             ['SHIELD FACE FULL ANTI FOG', 'plum'],
             ['RESP. PART. FILTER REG', 'darkviolet']]
        
        count=0
        for i, val in enumerate(col_labels2):
            ax.annotate('  '+val[0], xy=(hoffset + count * col_width[i], voffset),
            xycoords='axes fraction', ha='left', va='bottom', 
            rotation=-25, size=8, c=val[1])
            count+=1
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.1, hspace=1.1)
        