# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:03:46 2024

@author: busse
"""

import pandas as pd
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import skgstat as skg
from pykrige.ok import OrdinaryKriging


# Get the current script's path

script_path = os.path.abspath(sys.argv[0])

# Add the granada folder to the Python path
DQM_path = os.path.join(os.path.dirname(script_path), '..', 'DQM')
sys.path.append(DQM_path)

obs_xyz_path = os.path.join(DQM_path, 'OBS_XYZ_gdf.csv')

# Read OBS_XYZ.csv into a DataFrame
OBS_XYZ_gdf = pd.read_csv(obs_xyz_path)


# Save file in current folder for simplicity
OBS_XYZ_gdf.to_csv('OBS_XYZ_gdf.csv', index=False)

