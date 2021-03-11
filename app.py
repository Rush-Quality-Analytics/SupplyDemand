import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pathlib
import time
import app_fxns

import pandas as pd
import json

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

models = ['Logistic (multi-phase)', 'Gaussian (multi-phase)', 'Phase Wave', 'Time series analysis', 'Quadratic', 'Exponential']


######################## DASH APP FUNCTIONS ##################################




app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='COVID Calculator', children=[
        
        
        html.Div(
            id='df1', 
            style={'display': 'none'}
        ),
        
        html.Div(
            id='df2',
            style={'display': 'none'}
        ),

        html.Div(
            id='df3',
            style={'display': 'none'}
        ),
        
        
        
        
        # Banner
        html.Div(
            id="banner1",
            className="banner",
            children=[html.Img(src=app.get_asset_url("RUSH_full_color.jpg"), 
                               style={'textAlign': 'left'}),
                      html.Img(src=app.get_asset_url("plotly_logo.png"), 
                               style={'textAlign': 'right'})],
            style={#'width': '100%', 'display': 'inline-block',
                                 #'border-radius': '15px',
                                 #'box-shadow': '1px 1px 1px grey',
                                 'background-color': 'rgb(245, 247, 249)',
                                 #'padding': '10px',
                                 #'margin-bottom': '10px',
                                 #'fontSize':16
                            },
        ),
        # Left column
        html.Div(
            id="left-column1",
            className="three columns",
            children=[app_fxns.description_card1(), app_fxns.generate_control_card1()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside1", 
                    style={"display": "none"}
                )
            ],
            style={
                'border-radius': '15px',
                'box-shadow': '1px 1px 1px grey',
                'background-color': '#f0f0f0',
                'padding': '10px',
                'margin-bottom': '10px',
                #'fontSize':16
            },
        ),
        # Right column
        html.Div(
            id="right-column1",
            className="nine columns",
            children=[
                
                html.Div(
                        id="Figure1",
                        children=[dcc.Loading(
                            id="loading-1",
                            type="default",
                            fullscreen=False,
                            children=[
                                # Plot of model forecast
                                html.Div(
                                    id="model_forecasts1",
                                    children=[
                                        html.B("Model Forecasts. Exponential and Quadratic models run quickly." +
                                               " Other models are more intensive and may take several seconds."),
                                        html.Hr(),
                                        dcc.Graph(id="model_forecasts_plot1"),
                                    ],
                                    style={'border-radius': '15px',
                                           'box-shadow': '1px 1px 1px grey',
                                           'background-color': '#f0f0f0',
                                           'padding': '10px',
                                           'margin-bottom': '10px',
                                           'fontSize':16
                                            },
                                ),
                            ],
                        ),],),
                                
                
                html.A('Download CSV', id='model_forecast_link', download="model_forecast_data.csv",
                       href="",
                       target="_blank"),
                html.Br(),
                html.Br(),
                
                html.Div(
                        id="Table4",
                        children=[dcc.Loading(
                            id="loading-7",
                            type="default",
                            fullscreen=False,
                            children=[
                                html.Div(
                                    id="new_cases1",
                                    children=[
                                        html.B("New and Active Cases"),
                                        html.Hr(),
                                        dcc.Graph(id="new_cases_plot1"),
                                    ],
                                    style={'border-radius': '15px',
                                           'box-shadow': '1px 1px 1px grey',
                                           'background-color': '#f0f0f0',
                                           'padding': '10px',
                                           'margin-bottom': '10px',
                                           'fontSize':16
                                            },
                                ),
                            ],
                        ),],),
                
                html.Br(),
                html.Br(),
                
                html.Div(
                        id="Table1",
                        children=[dcc.Loading(
                            id="loading-2",
                            type="default",
                            fullscreen=False,
                            children=[
                                html.Div(
                                    id="patient_census1",
                                    children=[
                                        html.B("Forecasted Patient Census"),
                                        html.Hr(),
                                        dcc.Graph(id="patient_census_plot1"),
                                    ],
                                    style={'border-radius': '15px',
                                           'box-shadow': '1px 1px 1px grey',
                                           'background-color': '#f0f0f0',
                                           'padding': '10px',
                                           'margin-bottom': '10px',
                                           'fontSize':16
                                            },
                                ),
                            ],
                        ),],),
                
                html.Br(),
                html.Br(),
                
                html.Div(
                        id="Figure2",
                        children=[dcc.Loading(
                            id="loading-3",
                            type="default",
                            fullscreen=False,
                            children=[
                                html.Div(
                                    id="patient_discharge1",
                                    children=[
                                        html.B("Forecasted Patient Discharges"),
                                        html.Hr(),
                                        dcc.Graph(id="patient_discharge_plot1"),
                                    ],
                                    style={'border-radius': '15px',
                                           'box-shadow': '1px 1px 1px grey',
                                           'background-color': '#f0f0f0',
                                           'padding': '10px',
                                           'margin-bottom': '10px',
                                           'fontSize':16
                                            },
                                ),
                            ],
                        ),],),
                
                html.Br(),
                html.Br(),
                
                html.Div(
                        id="Table2",
                        children=[dcc.Loading(
                            id="loading-4",
                            type="default",
                            fullscreen=False,
                            children=[
                                html.Div(
                                    id="patient_census_table1",
                                    children=[
                                        html.B("Patient Census and Discharge Table"),
                                        html.Hr(),
                                        dcc.Graph(id="patient_census_table_plot1"),
                                    ],
                                    style={'border-radius': '15px',
                                           'box-shadow': '1px 1px 1px grey',
                                           'background-color': '#f0f0f0',
                                           'padding': '10px',
                                           'margin-bottom': '10px',
                                           'fontSize':16
                                            },
                                ),
                            ],
                        ),],),
                
                html.A('Download CSV', id='Patient_Census_Discharge_link', download="Patient_Census_Discharge_data.csv",
                       href="",
                       target="_blank"),
                html.Br(),
                html.Br(),
                
                
                html.Div(
                        id="Figure3",
                        children=[dcc.Loading(
                            id="loading-5",
                            type="default",
                            fullscreen=False,
                            children=[
                                html.Div(
                                    id="ppe1",
                                    children=[
                                        html.B("Forecasted PPE Needs"),
                                        html.Hr(),
                                        dcc.Graph(id="ppe_plot1"),
                                    ],
                                    style={'border-radius': '15px',
                                           'box-shadow': '1px 1px 1px grey',
                                           'background-color': '#f0f0f0',
                                           'padding': '10px',
                                           'margin-bottom': '10px',
                                           'fontSize':16
                                            },
                                ),
                            ],
                        ),],),
                
                
                html.Br(),
                html.Br(),
                
                html.Div(
                        id="Table3",
                        children=[dcc.Loading(
                            id="loading-6",
                            type="default",
                            fullscreen=False,
                            children=[
                                html.Div(
                                    id="ppe_table1",
                                    children=[
                                        html.B("PPE Forecast Table"),
                                        html.Hr(),
                                        dcc.Graph(id="ppe_table_plot1"),
                                    ],
                                    style={'border-radius': '15px',
                                           'box-shadow': '1px 1px 1px grey',
                                           'background-color': '#f0f0f0',
                                           'padding': '10px',
                                           'margin-bottom': '10px',
                                           'fontSize':16
                                            },
                                ),
                            ],
                        ),],),
                
                html.A('Download CSV', id='ppe_link', download="PPE_Forecast_data.csv",
                       href="",
                       target="_blank"),
                html.Br(),
                html.Br(),
                
            ],
        ),
        ],
        ),
        
        dcc.Tab(label='Employee Forecasts', children=[
            
        # Banner
        html.Div(
            id="banner4",
            className="banner",
            children=[html.Img(src=app.get_asset_url("RUSH_full_color.jpg"), 
                               style={'textAlign': 'left'}),
                      html.Img(src=app.get_asset_url("plotly_logo.png"), 
                               style={'textAlign': 'right'})],
            ),
            
            # Left column
            html.Div(
                id="left-column1b",
                className="three columns",
                children=[app_fxns.description_card1b(), app_fxns.generate_control_card2()]
                + [
                    html.Div(
                        ["initial child"], id="output-clientside1b", 
                        style={"display": "none"}
                    )
                ],
                style={
                    'border-radius': '15px',
                    'box-shadow': '1px 1px 1px grey',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
                    #'fontSize':16
                },
            ),
            
            # Right column
        html.Div(
            id="right-column1b",
            className="nine columns",
            children=[html.Div(
                        id="Table4b",
                        children=[dcc.Loading(
                            id="loading-7b",
                            type="default",
                            fullscreen=False,
                            children=[
                                html.Div(
                                    id="employee_cases1",
                                    children=[
                                        html.B("Forecasts of new and active cases among employees"),
                                        html.Hr(),
                                        dcc.Graph(id="employee_forecast_plot1"),
                                    ],
                                    style={'border-radius': '15px',
                                           'box-shadow': '1px 1px 1px grey',
                                           'background-color': '#f0f0f0',
                                           'padding': '10px',
                                           'margin-bottom': '10px',
                                           'fontSize':16
                                            },
                                ),
                            ],
                        ),],),
                ],
                ),
                
        ]),    
        
        
        
        
        
            
        dcc.Tab(label='Instructions & Details', children=[
        
        # Banner
        html.Div(
            id="banner5",
            className="banner",
            children=[html.Img(src=app.get_asset_url("RUSH_full_color.jpg"), 
                               style={'textAlign': 'left'}),
                      html.Img(src=app.get_asset_url("plotly_logo.png"), 
                               style={'textAlign': 'right'})],
            style={#'width': '100%', 'display': 'inline-block',
                                 #'border-radius': '15px',
                                 #'box-shadow': '1px 1px 1px grey',
                                 'background-color': 'rgb(245, 247, 249)',
                                 #'padding': '10px',
                                 #'margin-bottom': '10px',
                                 #'fontSize':16
                            },
        ),
        
        html.Div(
            id="ContactInfo-column",
            className="twelve columns",
            children=[app_fxns.description_card2()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside2", 
                    style={"display": "none"}
                )
            ],
            style={
                'border-radius': '15px',
                'box-shadow': '1px 1px 1px grey',
                'background-color': '#f0f0f0',
                'padding': '10px',
                'margin-bottom': '10px',
                #'fontSize':16
            },
        ),
        
        html.Div(
            id="Instructions-column",
            className="twelve columns",
            children=[app_fxns.description_card3()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside3", 
                    style={"display": "none"}
                )
            ],
            style={
                'border-radius': '15px',
                'box-shadow': '1px 1px 1px grey',
                'background-color': '#f0f0f0',
                'padding': '10px',
                'margin-bottom': '10px',
                #'fontSize':16
            },
        ),
        
        html.Div(
            id="Details-column",
            className="twelve columns",
            children=[app_fxns.description_card4()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside4", 
                    style={"display": "none"}
                )
            ],
            style={
                'border-radius': '15px',
                'box-shadow': '1px 1px 1px grey',
                'background-color': '#f0f0f0',
                'padding': '10px',
                'margin-bottom': '10px',
                #'fontSize':16
            },
        ),
        
        
        
        ]),
    ]),
])


#########################################################################################
################################ LOADING CALLBACKS ######################################
#########################################################################################



@app.callback(
    dash.dependencies.Output('ICU beds1-container', 'children'),
    [dash.dependencies.Input('ICU beds1', 'value')])
def update_output1(value):
    return 'ICU beds in house: {}'.format(value)


@app.callback(
    dash.dependencies.Output('nonICU beds1-container', 'children'),
    [dash.dependencies.Input('nonICU beds1', 'value')])
def update_output2(value):
    return 'non-ICU beds in house: {}'.format(value)


@app.callback(
    dash.dependencies.Output('visits1-container', 'children'),
    [dash.dependencies.Input('visits1', 'value')])
def update_output3(value):
    return '% of new cases visiting to your hospital: {}'.format(value)


@app.callback(
    dash.dependencies.Output('admits1-container', 'children'),
    [dash.dependencies.Input('admits1', 'value')])
def update_output4(value):
    return '% of visits admitted: {}'.format(value)


@app.callback(
    dash.dependencies.Output('percent ICU1-container', 'children'),
    [dash.dependencies.Input('percent ICU1', 'value')])
def update_output5(value):
    return '% of visits admitted to ICU: {}'.format(value)


@app.callback(
    dash.dependencies.Output('transfers1-container', 'children'),
    [dash.dependencies.Input('transfers1', 'value')])
def update_output6(value):
    return 'Daily number of transfers admitted: {}'.format(value)


@app.callback(
    dash.dependencies.Output('percent transferICU1-container', 'children'),
    [dash.dependencies.Input('percent transferICU1', 'value')])
def update_output7(value):
    return '% of transfers admitted to ICU: {}'.format(value)


@app.callback(
    dash.dependencies.Output('on vent1-container', 'children'),
    [dash.dependencies.Input('on vent1', 'value')])
def update_output8(value):
    return '% of ICU patients on ventilators: {}'.format(value)


@app.callback(
    dash.dependencies.Output('non-ICU LOS1-container', 'children'),
    [dash.dependencies.Input('non-ICU LOS1', 'value')])
def update_output9(value):
    return 'non-ICU length of stay: {}'.format(value)


@app.callback(
    dash.dependencies.Output('ICU LOS1-container', 'children'),
    [dash.dependencies.Input('ICU LOS1', 'value')])
def update_output10(value):
    return 'ICU length of stay: {}'.format(value)


@app.callback(
    dash.dependencies.Output('mortality1-container', 'children'),
    [dash.dependencies.Input('mortality1', 'value')])
def update_output11(value):
    return 'ICU mortality rate: {}'.format(value)


@app.callback(
    dash.dependencies.Output('time lag1-container', 'children'),
    [dash.dependencies.Input('time lag1', 'value')])
def update_output12(value):
    return 'Time lag in hospital visitation: {}'.format(value)






@app.callback(
    dash.dependencies.Output('ICU beds2-container', 'children'),
    [dash.dependencies.Input('ICU beds2', 'value')])
def update_output2_1(value):
    return 'ICU beds in house: {}'.format(value)


@app.callback(
    dash.dependencies.Output('nonICU beds2-container', 'children'),
    [dash.dependencies.Input('nonICU beds2', 'value')])
def update_output2_2(value):
    return 'non-ICU beds in house: {}'.format(value)


@app.callback(
    dash.dependencies.Output('vents in house2-container', 'children'),
    [dash.dependencies.Input('vents in house2', 'value')])
def update_output2_3(value):
    return 'Ventilators in house: {}'.format(value)


@app.callback(
    dash.dependencies.Output('visits2-container', 'children'),
    [dash.dependencies.Input('visits2', 'value')])
def update_output2_4(value):
    return '% visits to your hospital: {}'.format(value)


@app.callback(
    dash.dependencies.Output('admits2-container', 'children'),
    [dash.dependencies.Input('admits2', 'value')])
def update_output2_5(value):
    return '% of visits admitted: {}'.format(value)


@app.callback(
    dash.dependencies.Output('percent ICU2-container', 'children'),
    [dash.dependencies.Input('percent ICU2', 'value')])
def update_output2_6(value):
    return '% of visits admitted to ICU: {}'.format(value)


@app.callback(
    dash.dependencies.Output('transfers2-container', 'children'),
    [dash.dependencies.Input('transfers2', 'value')])
def update_output2_7(value):
    return 'Daily number of transfers admitted: {}'.format(value)


@app.callback(
    dash.dependencies.Output('percent transferICU2-container', 'children'),
    [dash.dependencies.Input('percent transferICU2', 'value')])
def update_output2_8(value):
    return '% of transfers admitted to ICU: {}'.format(value)


@app.callback(
    dash.dependencies.Output('on vent2-container', 'children'),
    [dash.dependencies.Input('on vent2', 'value')])
def update_output2_9(value):
    return '% of ICU patients on ventilators: {}'.format(value)


@app.callback(
    dash.dependencies.Output('non-ICU LOS2-container', 'children'),
    [dash.dependencies.Input('non-ICU LOS2', 'value')])
def update_output2_10(value):
    return 'non-ICU length of stay: {}'.format(value)


@app.callback(
    dash.dependencies.Output('ICU LOS2-container', 'children'),
    [dash.dependencies.Input('ICU LOS2', 'value')])
def update_output2_11(value):
    return 'ICU length of stay: {}'.format(value)


@app.callback(
    dash.dependencies.Output('mortality2-container', 'children'),
    [dash.dependencies.Input('mortality2', 'value')])
def update_output2_12(value):
    return 'ICU mortality rate: {}'.format(value)


@app.callback(
    dash.dependencies.Output('time lag2-container', 'children'),
    [dash.dependencies.Input('time lag2', 'value')])
def update_output2_13(value):
    return 'Time lag in hospital visitation: {}'.format(value)





@app.callback(
    dash.dependencies.Output('GLOVE SURGICAL-container', 'children'),
    [dash.dependencies.Input('gloves1', 'value')])
def update_output2_14(value):
    return 'GLOVE SURGICAL: {}'.format(value)


@app.callback(
    dash.dependencies.Output('GLOVE EXAM NITRILE-container', 'children'),
    [dash.dependencies.Input('gloves2', 'value')])
def update_output2_15(value):
    return 'GLOVE EXAM NITRILE: {}'.format(value)


@app.callback(
    dash.dependencies.Output('GLOVE EXAM VINYL-container', 'children'),
    [dash.dependencies.Input('gloves3', 'value')])
def update_output2_16(value):
    return 'GLOVE EXAM NITRILE: {}'.format(value)


@app.callback(
    dash.dependencies.Output('MASK FACE PROC ANTI FOG-container', 'children'),
    [dash.dependencies.Input('mask1', 'value')])
def update_output2_17(value):
    return 'MASK FACE PROC ANTI FOG: {}'.format(value)


@app.callback(
    dash.dependencies.Output('MASK PROC FLUID RESISTANT-container', 'children'),
    [dash.dependencies.Input('mask2', 'value')])
def update_output2_18(value):
    return 'MASK PROC FLUID RESISTANT: {}'.format(value)


@app.callback(
    dash.dependencies.Output('GOWN ISOLATION XL YELLOW-container', 'children'),
    [dash.dependencies.Input('gown1', 'value')])
def update_output2_19(value):
    return 'GOWN ISOLATION XL YELLOW: {}'.format(value)


@app.callback(
    dash.dependencies.Output('MASK SURG ANTI FOG W/FILM-container', 'children'),
    [dash.dependencies.Input('mask3', 'value')])
def update_output2_20(value):
    return 'MASK SURG ANTI FOG W/FILM: {}'.format(value)


@app.callback(
    dash.dependencies.Output('SHIELD FACE FULL ANTI FOG-container', 'children'),
    [dash.dependencies.Input('shield1', 'value')])
def update_output2_21(value):
    return 'SHIELD FACE FULL ANTI FOG: {}'.format(value)


@app.callback(
    dash.dependencies.Output('RESP PART FILTER REG-container', 'children'),
    [dash.dependencies.Input('resp1', 'value')])
def update_output2_22(value):
    return 'RESP PART FILTER REG: {}'.format(value)


@app.callback(
    dash.dependencies.Output('incidence rate-container', 'children'),
    [dash.dependencies.Input('inc_rate', 'value')])
def update_output2_23(value):
    return 'COVID positives among your employees is {}'.format(value) + '% of that for the general population'



@app.callback( # Select sub-category
    Output('county-select1', 'value'),
    [
     Input('county-select1', 'options'),
     Input('location-select1', 'value'),
     ],
    )
def update_output15(available_options, v2):
    return available_options[0]['value']


@app.callback( # Select sub-category
    Output('county-select2', 'value'),
    [
     Input('county-select2', 'options'),
     Input('location-select2', 'value'),
     ],
    )
def update_output18(available_options, v2):
    return available_options[0]['value']



@app.callback( # Select sub-category
    Output('location-select1', 'value'),
    [
     Input('location-select1', 'options'),
     #Input('county-select1', 'value'),
     ],
    )
def update_output16(available_options):
    return available_options[0]['value']


@app.callback( # Select sub-category
    Output('location-select2', 'value'),
    [
     Input('location-select2', 'options'),
     #Input('county-select1', 'value'),
     ],
    )
def update_output19(available_options):
    return available_options[0]['value']



@app.callback( # Update available sub_categories
    Output('county-select1', 'options'),
    [
     Input('location-select1', 'value'),
     #Input('county-select1', 'value'),
     ],
    )
def update_output13(v1):
    counties_df = []
    with open('DataUpdate/data/COVID-CASES-Counties-DF.txt.gz', 'rb') as csvfile:
                counties_df = pd.read_csv(csvfile, compression='gzip', sep='\t')
    counties_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    counties_df = counties_df[~counties_df['Admin2'].isin(['Unassigned', 'Out-of-state', 
                                                       'Out of AL', 'Out of IL',
                                                       'Out of CO', 'Out of GA',
                                                       'Out of HI', 'Out of LA',
                                                       'Out of ME', 'Out of MI',
                                                       'Out of OK', 'Out of PR',
                                                       'Out of TN', 'Out of UT',
                                                       ])]
    
    tdf = counties_df[counties_df['Province/State'] == v1]
    counties_df = 0
    cts = sorted(list(set(tdf['Admin2'].values.tolist())))
    tdf = 0
    l = 'Entire state or territory'
    cts.insert(0, l)
    return [{"label": i, "value": i} for i in cts]



@app.callback( # Update available sub_categories
    Output('county-select2', 'options'),
    [
     Input('location-select2', 'value'),
     #Input('county-select1', 'value'),
     ],
    )
def update_output20(v1):
    counties_df = []
    with open('DataUpdate/data/COVID-CASES-Counties-DF.txt.gz', 'rb') as csvfile:
                counties_df = pd.read_csv(csvfile, compression='gzip', sep='\t')
    counties_df.drop(['Unnamed: 0'], axis=1, inplace=True) 
    
    counties_df = counties_df[~counties_df['Admin2'].isin(['Unassigned', 'Out-of-state', 
                                                       'Out of AL', 'Out of IL',
                                                       'Out of CO', 'Out of GA',
                                                       'Out of HI', 'Out of LA',
                                                       'Out of ME', 'Out of MI',
                                                       'Out of OK', 'Out of PR',
                                                       'Out of TN', 'Out of UT',
                                                       ])]
    
    tdf = counties_df[counties_df['Province/State'] == v1]
    counties_df = 0
    cts = sorted(list(set(tdf['Admin2'].values.tolist())))
    tdf = 0
    l = 'Entire state or territory'
    cts.insert(0, l)
    return [{"label": i, "value": i} for i in cts]




@app.callback( # Update available sub_categories
    Output('model-select1', 'options'),
    [
     Input('location-select1', 'value'),
     Input('county-select1', 'value'),
     ],
    )
def update_output14(loc1, loc2):
    return [{"label": i, "value": i} for i in models]


@app.callback( # Update available sub_categories
    Output('model-select2', 'options'),
    [
     Input('location-select2', 'value'),
     Input('county-select2', 'value'),
     ],
    )
def update_output17(loc1, loc2):
    return [{"label": i, "value": i} for i in models]




@app.callback(
     [Output('df1', 'children'), 
      Output("model_forecasts_plot1", "figure")],
     [Input("location-select1", "value"),
      Input("county-select1", "value"),
      Input("model-select1", "value"),
      #Input("add-forecast1", "n_clicks"),
      Input("reset-btn1", "n_clicks")
     ],
)
def update_model_forecast1(loc, county, model, reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    df_fits = app_fxns.generate_model_forecasts(loc, county, model, reset)
    fig1 = app_fxns.generate_model_forecast_plot(df_fits, reset)
    
    return df_fits, fig1



@app.callback(
      Output("new_cases_plot1", "figure"),
     [Input('df1', 'children'), 
      Input("location-select1", "value"),
      Input("county-select1", "value"),
      Input("reset-btn1", "n_clicks")
     ],
)
def update_model_forecast111(df, loc, county, reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    fig = app_fxns.generate_plot_new_cases(df, loc, county, reset)
    
    return fig



@app.callback(
    Output('model_forecast_link', 'href'),
    [Input('df1', 'children'),
     Input("location-select1", "value"),
     Input('county-select1', 'value'),
     Input("reset-btn1", "n_clicks")],
)
def update_table_model_forecast1(df_fits, loc, cty, reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.generate_model_forecast_table(df_fits, reset)




@app.callback(
    Output('df2', 'children'),
    [Input("location-select1", "value"),
     Input('county-select1', 'value'),
     Input("model-select1", "value"),
     Input("ICU beds1", "value"),
     Input("nonICU beds1", "value"),
     Input("visits1",  "value"),
     Input("admits1",  "value"),
     Input("percent ICU1", "value"),
     Input("ICU LOS1", "value"),
     Input("non-ICU LOS1", "value"),
     Input("on vent1",  "value"),
     Input("time lag1", "value"),
     Input("transfers1", "value"),
     Input("percent transferICU1", "value"),
     Input('mortality1', "value"),
     Input("gloves1", "value"),
     Input("gloves2", "value"),
     Input("gloves3", "value"),
     Input("mask1", "value"),
     Input("mask2",  "value"),
     Input("gown1", "value"),
     Input("mask3", "value"),
     Input("shield1", "value"),
     Input('resp1', "value"),
     Input("reset-btn1", "n_clicks"),
    ],
)


def update_patient_census(loc, cty, model, icu_beds, nonicu_beds, per_loc, per_admit, 
    per_cc, LOS_cc, LOS_nc, per_vent, TimeLag, transfers, per_ICU_transfer, mortality, 
    GLOVE_SURGICAL, GLOVE_EXAM_NITRILE, GLOVE_EXAM_VINYL, MASK_FACE_PROC_ANTI_FOG, 
    MASK_PROC_FLUID_RESISTANT, GOWN_ISOLATION_XL_YELLOW, MASK_SURG_ANTI_FOG_FILM, 
    SHIELD_FACE_FULL_ANTI_FOG, RESP_PART_FILTER_REG, reset_click):

    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True
            
    # Return to original hm(no colored annotation) by resetting
    
    df2 = app_fxns.generate_patient_census(loc, cty, model, icu_beds, nonicu_beds, per_loc, per_admit, 
    per_cc, LOS_cc, LOS_nc, per_vent, TimeLag, transfers, per_ICU_transfer, mortality, 
    GLOVE_SURGICAL, GLOVE_EXAM_NITRILE, GLOVE_EXAM_VINYL, MASK_FACE_PROC_ANTI_FOG, 
    MASK_PROC_FLUID_RESISTANT, GOWN_ISOLATION_XL_YELLOW, MASK_SURG_ANTI_FOG_FILM, 
    SHIELD_FACE_FULL_ANTI_FOG, RESP_PART_FILTER_REG, reset)
    
    return df2



@app.callback(
    Output("patient_census_plot1", "figure"),
    [Input('df2', 'children'),
     Input("location-select1", "value"),
     Input('county-select1', 'value'),
     Input("reset-btn1", "n_clicks")],
)

def update_plot_patient_census(df_census, loc, cty, reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.generate_plot_patient_census(df_census, reset)



@app.callback(
     Output('df3', 'children'),
     [Input("location-select2", "value"),
      Input("county-select2", "value"),
      Input("model-select2", "value"),
      Input("reset-btn2", "n_clicks")
     ],
)
def update_model_forecast2(loc, county, model, reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn2":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    df_fits = app_fxns.generate_model_forecasts(loc, county, model, reset)
    
    return df_fits


@app.callback(
    Output("employee_forecast_plot1", "figure"),
    [Input('df3', 'children'),
     Input("location-select2", "value"),
     Input('county-select2', 'value'),
     Input("employees", "value"),
     Input("inc_rate", "value"),
     Input("furlough", "value"),
     Input("reset-btn2", "n_clicks")],
)

def update_plot_employee_forecast(df_census, loc, cty, employees, inc_rate, furlough, reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn2":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.generate_plot_employee_forecast1(df_census, loc, cty, employees, inc_rate, furlough, reset)




@app.callback(
    [Output("patient_census_table_plot1", "figure"), Output('Patient_Census_Discharge_link', 'href')],
    [Input('df2', 'children'),
     Input("location-select1", "value"),
     Input('county-select1', 'value'),
     Input("reset-btn1", "n_clicks")],
)
def update_table_patient_census1(df_census, loc, cty, reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.generate_patient_census_table(df_census, reset)


@app.callback(
    Output("patient_discharge_plot1", "figure"),
    [Input('df2', 'children'),
     Input("location-select1", "value"),
     Input('county-select1', 'value'),
     Input("reset-btn1", "n_clicks")],
)

def update_plot_patient_discharge(df_census, loc, cty, reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.generate_plot_discharge_census(df_census, reset)






@app.callback(
    Output("ppe_plot1", "figure"),
    [Input('df2', 'children'),
     Input("location-select1", "value"),
     Input('county-select1', 'value'),
     Input("reset-btn1", "n_clicks")],
)

def update_plot_ppe(df, loc, cty, reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.generate_plot_ppe(df, reset)




@app.callback(
    [Output("ppe_table_plot1", "figure"), Output('ppe_link', 'href')],
    [Input('df2', 'children'),
     Input("location-select1", "value"),
     Input('county-select1', 'value'),
     Input("reset-btn1", "n_clicks")],
)
def update_table_ppe(df, loc, cty, reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.generate_ppe_table(df, reset)











@app.callback(
    Output("testing_rate_map", "figure"),
    [Input("reset-btn1", "n_clicks")],
)
def update_map1(reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.map1(reset)


@app.callback(
    Output("positive_tests_per_capita_map", "figure"),
    [Input("reset-btn1", "n_clicks")],
)
def update_map2(reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.map2(reset)


@app.callback(
    Output("percent_positive_tests_map", "figure"),
    [Input("reset-btn1", "n_clicks")],
)
def update_map3(reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.map3(reset)


@app.callback(
    Output("testing_rate_change_map", "figure"),
    [Input("reset-btn1", "n_clicks")],
)
def update_map4(reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.map4(reset)



@app.callback(
    Output("hospitilization_rate_map", "figure"),
    [Input("reset-btn1", "n_clicks")],
)
def update_map5(reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.map5(reset)



@app.callback(
    Output("cumulative_hospitalizations_map", "figure"),
    [Input("reset-btn1", "n_clicks")],
)
def update_map6(reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.map6(reset)




@app.callback(
    Output("delta_testing_rate_plot", "figure"),
    [Input("reset-btn1", "n_clicks")],
)
def update_generate_delta_testing_plot(reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.generate_delta_testing_plot(reset)



@app.callback(
    Output("generate_PopSize_vs_Tested", "figure"),
    [Input("reset-btn1", "n_clicks")],
)
def update_generate_PopSize_vs_Tested_plot(reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.generate_PopSize_vs_Tested(reset)


@app.callback(
    Output("generate_negative_vs_tested", "figure"),
    [Input("reset-btn1", "n_clicks")],
)
def update_generate_Negative_vs_Tested_plot(reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.generate_Negative_vs_Tested(reset)


@app.callback(
    Output("generate_Positive_vs_Tested", "figure"),
    [Input("reset-btn1", "n_clicks")],
)
def update_generate_Positive_vs_Tested_plot(reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.generate_Positive_vs_Tested(reset)



@app.callback(
    Output("generate_ICU_vs_Hospitalized", "figure"),
    [Input("reset-btn1", "n_clicks")],
)
def update_generate_ICU_vs_Hospitalized_plot(reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.generate_ICU_vs_Hospitalized(reset)


@app.callback(
    Output("generate_ventilator_vs_ICU", "figure"),
    [Input("reset-btn1", "n_clicks")],
)
def update_generate_ventilator_vs_ICU_plot(reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn1":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return app_fxns.generate_ventilator_vs_ICU(reset)




# Run the server
if __name__ == "__main__":
    app.run_server(host='127.0.0.1', port=8050, debug=True, threaded=False, processes=11)