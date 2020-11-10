import numpy as np
import pandas as pd
import datetime
from scipy import stats
import sys
import re

counties_df = pd.read_csv('DataUpdate/data/COVID-CASES-Counties-DF.txt', sep='\t') 
counties_df = counties_df[~counties_df['Admin2'].isin(['Unassigned', 'Out-of-state', 
                                'Out of AL', 'Out of IL','Out of CO', 'Out of GA',
                                'Out of HI', 'Out of LA', 'Out of ME', 'Out of MI',
                                'Out of OK', 'Out of PR', 'Out of TN', 'Out of UT',])]


cty_pops = pd.read_excel('DataUpdate/data/co-est2019-annres.xlsx')
cty_pops = cty_pops[cty_pops['Geographic Area'] != 'United States']


counties = []
states = []
for c in cty_pops['Geographic Area'].values.tolist():
    
    c = c.replace('.', '')
    
    county, state = c.split(", ", 1)
    
    try:
        county = county.replace('County', '')
    except:
        pass
    
    try:
        county = county.replace('county', '')
    except:
        pass
    
    try:
        county = county.replace('Parish', '')
    except:
        pass
    
    try:
        county = county.replace('parish', '')
    except:
        pass
    
    county = re.sub('\s$', '', county)
    state = re.sub('\s$', '', state)
    
    #if state == 'Illinois':
    #    print(county, state)
    
    counties.append(county)
    states.append(state)
    


cty_pops['County'] = counties
cty_pops['State'] = states
cty_pops['Population size'] = cty_pops[2019].values.tolist()

cty_pops.drop(['Geographic Area'], axis=1, inplace=True)
cty_pops.drop([2019], axis=1, inplace=True)

cty_pops.to_pickle('DataUpdate/data/County_Pops.pkl')
