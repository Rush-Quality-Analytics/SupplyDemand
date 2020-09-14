import matplotlib.pyplot as plt
import pandas as pd
import csv # csv import functionality
import time # library for time functionality
import sys
import warnings # needed for suppression of unnecessary warnings
import base64 # functionality for encoding binary data to ASCII characters and decoding back to binary data
import numpy as np
import datetime 
import sys
from math import pi


def fig_fxn(fig, model, n):
    
    fig.add_subplot(3, 3, n)
    Y = []
    
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Cumulative cases', fontsize=14)
    
    if model == 'Saturating models':
        x = np.linspace(-4, 5, 100)
        Y = 1/(1 + np.exp(-x + 1))
        plt.plot(x, Y, c ='k', linewidth=3, label = 'Logistic, Gaussian\n& SEIR-SD')

        
        
    elif model == 'Rapidly increasing models':
        x = np.linspace(0, 10, 100)
        r = 0.5
        Y = np.exp(r*x)
        plt.plot(x, Y, c ='0.6', linewidth=3, label='Exponential')
        
        x = np.linspace(0, 10, 100)
        Y = x**2 + x
        plt.plot(x, Y, c ='k', linewidth=3, label='Quadratic')
        
        
        
    elif model == '2-phase models': 
        x1 = np.linspace(-4, 5, 100)
        Y1 = 1/(1 + np.exp(-x1 + 1))
        
        x2 = x1 + 10
        Y2 = Y1 + np.max(Y1)
        
        x1 = x1.tolist()
        x1.extend(x2.tolist())
        Y1 = Y1.tolist()
        Y1.extend(Y2.tolist())
    
        
        plt.plot(x1, Y1, c ='k', linewidth=3, label='2-phase logistic')
        
        
        x1 = np.linspace(-4, 5, 100)
        c = 1
        b = 1
        f = -1.
        g = 1
        Y1 = 1 / (1 + np.exp(-c * (x1 + g*np.sin(f*x1)) + b))
        
        x2 = x1 + 10
        Y2 = Y1 + np.max(Y1)
        
        x1 = x1.tolist()
        x1.extend(x2.tolist())
        Y1 = Y1.tolist()
        Y1.extend(Y2.tolist())
    
        
        plt.plot(x1, Y1, c ='0.6', linewidth=3, label='2-phase sine-logistic')
    
    plt.legend(loc=2, fontsize=8, frameon=False)
        
    plt.title(model, fontweight='bold', fontsize=10)
    plt.yticks([])
    plt.xticks([])
    
    #plt.tick_params(axis='both', labelsize=8, rotation=45)
        
    
    
    return fig
    
    


fig = plt.figure(figsize=(10, 10))
models = ['Rapidly increasing models', 'Saturating models',
          '2-phase models']


ns = [1,2,3]
for i, model in enumerate(models):
    fig = fig_fxn(fig, model, ns[i])
    




plt.subplots_adjust(wspace=0.35, hspace=0.35)
plt.savefig('Model_Forms.png', dpi=400, bbox_inches = "tight")
plt.close()