# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 12:01:54 2021

@author: johnt
"""

import os
import sys
abspath = os.path.abspath(sys.argv[0])
dname = os.path.dirname(abspath)
os.chdir(dname)

from assignment_B_model import logistic
from assignment_B_model import calibration

import numpy as np
import pandas as panda
import matplotlib.pyplot as plt
import scipy.optimize as scipy 
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#%% Import Data

Goal611_data = panda.read_excel("Goal_6.1.1_All_Countries.xlsx","Goal6")

All_area_data = Goal611_data[~Goal611_data['Location'].isin(['URBAN','RURAL'])]

Empty = All_area_data.set_index('GeoAreaName')['Value'].isna().sum(level=0)
Max = All_area_data.groupby('GeoAreaName')['Value'].max()
Min = All_area_data.groupby('GeoAreaName')['Value'].min()
Average = All_area_data.groupby('GeoAreaName')['Value'].mean()
CountryName = All_area_data['GeoAreaName'].unique().tolist()


#%%Input Country
#modify this area to a selection box of 2 countries
country1data = All_area_data[(All_area_data['GeoAreaName'] == 'Afghanistan')]
"""country2data = All_area_data[(All_area_data['GeoAreaName'] == 'Spain')]"""

#%% Logistics Calibration
#use the loop for 2 countries
CalibrationModel = calibration(country1data["TimePeriod"],country1data["Value"])
x= CalibrationModel[0]
start= 0
K = CalibrationModel[1]
x_peak = CalibrationModel[2]
r = CalibrationModel[3]
LogisticsModel = logistic(x, start, K, x_peak, r)

"""
"Calculating Percentage Bias and the Root Mean Square Error"

PBIAS = 100* (sum(LogisticsModel - country1data["Value"])/sum(country1data["Value"]))
RMSE = (np.sqrt(((LogisticsModel - country1data["Value"]) ** 2).mean()))

cv = KFold(n_splits=10, random_state=1, shuffle=True) 
scores = cross_val_score(LogisticsModel, All_area_data["TimePeriod"], All_area_data["Value"], scoring='accuracy', cv=cv, n_jobs=-1)
"""



 


