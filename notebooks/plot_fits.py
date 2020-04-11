import ipywidgets as widgets # provides interactive functionality
import matplotlib.pyplot as plt # plotting library
import pandas as pd # data frame library

import numpy as np # numerical python


#### Define the class App_GetFits
#### Will contain all other functions for modeling, calculation, and plotting

class App_PlotFits:
    
    # Dataframe containing data aggregated from Johns Hopkins daily reports
    
    
    #### delcare objects intended to be used as global variables 

    
    # declare the following as global so they can be shared between functions
    # and classes
    
    
    def __init__(self, model_fits_df):
        
        self._model_fits_df = model_fits_df
        
        
        # declare widgets: dropdowns, floattexts, toggle buttons, datepicker, etc.
        self._1_toggle = self._create_toggle()
        
        self._2_floattext = self._create_floattext(label = 'Forecast length (days)', 
                                                    val=10, minv=1, maxv=60, boxw='33%', desw='70%')
        
        
        # define containers to hold the widgets, plots, and additional outputs
        self._plot_container = widgets.Output()
        
        _app_container = widgets.VBox(
            [widgets.VBox([widgets.HBox([self._1_toggle, self._2_floattext], 
                             layout=widgets.Layout(align_items='flex-start', flex='0 auto auto', width='100%'))],
                           
                           
                           layout=widgets.Layout(display='flex', flex_flow='column', border='solid 1px', 
                                        align_items='stretch', width='100%')),
                           
                           self._plot_container], layout=widgets.Layout(display='flex', flex_flow='column', 
                                        border='solid 2px', align_items='initial', width='100%'))
                
        # 'flex-start', 'flex-end', 'center', 'baseline', 'stretch', 'inherit', 'initial', 'unset'
        self.container = widgets.VBox([
            widgets.HBox([
                _app_container
            ])
        ], layout=widgets.Layout(align_items='flex-start', flex='auto', width='100%'))
        self._update_app()
        
        
    @classmethod
    def generate_plot(cls, model_fits_df):
        
        return cls(model_fits_df)
        
        
    
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
    
    
    
    
    def _on_change(self, _):
        # do the following when app inputs change
        self._update_app()

    def _update_app(self):
        # update the app when called
        
        # redefine input/parameter values
        log_scl = self._1_toggle.value
        ForecastDays = self._2_floattext.value
        
        
        # wait to clear the plots/tables until new ones are generated
        self._plot_container.clear_output(wait=True)
        
        with self._plot_container:
            # Run the functions to generate figures and tables
            self._plot_fit(log_scl, self._model_fits_df, ForecastDays)
            
            plt.show()
            
            
    def _plot_fit(self, log_scl, fits_df, ForecastDays):
        
        
        # declare figure object
        fig = plt.figure(figsize=(16, 6))
        # use subplot2grid functionality
        ax = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=4)
        
        labels = fits_df['label'].tolist()
        
        for i, label in enumerate(labels):
            
            sub_df = fits_df[fits_df['label'] == label]
            
            fdates = sub_df['forecast_dates'].iloc[0]
            forecasted_y = sub_df['forecasted_y'].iloc[0]
            clr = sub_df['fore_clr'].iloc[0]
            focal_loc = sub_df['focal_loc'].iloc[0]
            popsize = sub_df['PopSize'].iloc[0]
            
            pred_y = sub_df['pred_y'].iloc[0]
            # plot forecasted y values vs. dates
            l = int(len(pred_y)+ForecastDays)
            
            forecasted_y = forecasted_y[0 : l]
            fdates = fdates[0 : l]
            
            plt.plot(fdates, forecasted_y, c=clr, linewidth=3, label=label)
            
            
            dates = sub_df['pred_dates'].iloc[0]
            clr = sub_df['pred_clr'].iloc[0]
            obs_y = sub_df['obs_y'].iloc[0]
            
            # plot predicted y values vs. dates
            plt.plot(dates, pred_y, c=clr, linewidth=3)
            
            plt.scatter(dates, obs_y, c='0.2', s=100, alpha=0.8, linewidths=0.1)
            
        
        # For new cases, subtract today's number from yesterday's
        # in the case of negative values, reassign as 0
        
        forecast_df = fits_df[fits_df['label'] == 'Current forecast']
        forecast_vals = forecast_df['forecasted_y'].iloc[0]
        forecasted_x = list(range(len(forecast_vals)))
        obs_pred_r2 = forecast_df['obs_pred_r2'].iloc[0]
        model = forecast_df['model'].iloc[0]
        
        
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
        mod = str(model)
        
        if mod == 'polynomial':
            mod = '2nd degree polynomial'
            
        t_label = 'Forecasted cases for ' + focal_loc + '. Population size: ' + f"{popsize:,}"
        t_label += '\nPredicted via fitting the ' + mod + ' model'
        t_label += '. Current forecast ' + r'$r^{2}$' + ' = ' + str(np.round(obs_pred_r2, 3)) 
        plt.title(t_label, fontsize = 16, fontweight = 'bold')
        
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
        