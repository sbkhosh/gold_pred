#!/usr/bin/python3

import csv
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import pandas as pd 
import warnings
import yaml
import csv
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import pandas as pd 
import warnings
import yaml

from datetime import datetime, timedelta
from dt_help import Helper
from dt_read import DataProcessor
from dt_model import MLPrediction
from pandas.plotting import register_matplotlib_converters

warnings.filterwarnings('ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None 
register_matplotlib_converters()

if __name__ == '__main__':
    obj_helper = Helper('data_in','conf_help.yml')
    obj_helper.read_prm()
    
    fontsize = obj_helper.conf['font_size']
    matplotlib.rcParams['axes.labelsize'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = fontsize
    matplotlib.rcParams['ytick.labelsize'] = fontsize
    matplotlib.rcParams['legend.fontsize'] = fontsize
    matplotlib.rcParams['axes.titlesize'] = fontsize
    matplotlib.rcParams['text.color'] = 'k'

    obj_0 = DataProcessor('data_in','data_out','conf_model.yml')
    obj_0.read_prm()   
    obj_0.read_tickers()
    obj_0.process()

    ml_pred = MLPrediction(data=obj_0.values,
                           yvar=obj_0.conf.get('yvar'),
                           mpg=obj_0.conf.get('mapping'),
                           num_selected_models=obj_0.conf.get('num_selected_models'))
    
    # features
    ml_pred.get_returns()
    ml_pred.get_mov_avg()
    ml_pred.get_features()
    
    # label
    ml_pred.get_target()
    ml_pred.process_all()

    # regression modeling
    ml_pred.regr_models()
    
    # tuning best three models
    ml_pred.get_best_models()

    # tune/bagging best models
    ml_pred.bagg_tune_best_model()

    # stacking models
    ml_pred.stacking_model()

    # saving model
    ml_pred.save_model()

    # predicting model
    ml_pred.predict()
    
