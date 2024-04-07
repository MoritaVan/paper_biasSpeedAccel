#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 12:22:01 2020

@author: Vanessa Morita


"""
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("../functions")
from functions.utils import *
from functions.updateRC import fontsizeDict,rcConfigDict

plt.rcParams.update(fontsizeDict(small=10,medium=12))
plt.rcParams.update(rcConfigDict(filepath = "../rcparams_config.json"))

cm = 1/2.54  # centimeters in inches
single_col = 9*cm
# oneDot5_col = 12.7*cm
two_col = 19*cm

main_dir = "../data/biasAcceleration"
os.chdir(main_dir)

import warnings
warnings.filterwarnings("ignore")
import traceback

output_folder = '../outputs/exp2'