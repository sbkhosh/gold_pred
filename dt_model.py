#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from dt_help import Helper
from pycaret.regression import *

class MLPrediction():
    def __init__(
        self, 
        data: pd.DataFrame,
        yvar: 'GC=F',
        mpg: dict,
        num_selected_models: int
    ):
        self.data = data 
        self.cols = self.data.columns.values
        self.feat_days = [1,3,5] # ,5,14,21,60,90,180,250
        self.trgt_days = [14,21]
        self.yvar = yvar
        self.to_blacklist = ['lr','en','lasso','ridge','lar','llar','omp','br','ard','par',
                             'ransac','tr','huber','kr','dt','rf','ada','knn','xgboost','mlp','lightgbm','svm']
        self.mpg = mpg
        self.num_selected_models = num_selected_models
        
        
    @Helper.timing
    def get_returns(self):
        data_temp = []
        cols = self.data.columns.drop('Dates')
        for i in self.feat_days:
            data_temp.append(self.data[cols].pct_change(periods=i).add_suffix("_"+str(i)+'D'))
        self.returns = pd.concat([self.data]+data_temp,axis=1)
        
    @Helper.timing
    def get_mov_avg(self):
        yvar = self.yvar
        moving_avg = pd.DataFrame(self.data['Dates'],columns=['Dates'])
        moving_avg['Dates']=pd.to_datetime(moving_avg['Dates'],format='%Y-%b-%d')
        moving_avg[yvar+'_180EWMA'] = (self.data[yvar]/(self.data[yvar].ewm(span=180,adjust=True,ignore_na=False).mean()))-1
        moving_avg[yvar+'_90EWMA'] = (self.data[yvar]/(self.data[yvar].ewm(span=90,adjust=True,ignore_na=True).mean()))-1
        moving_avg[yvar+'_60EWMA'] = (self.data[yvar]/(self.data[yvar].ewm(span=60,adjust=True,ignore_na=True).mean()))-1
        moving_avg[yvar+'_30EWMA'] = (self.data[yvar]/(self.data[yvar].ewm(span=30,adjust=True,ignore_na=True).mean()))-1
        self.mov_avg = moving_avg

    @Helper.timing
    def get_features(self):
        self.features = Helper.get_all_features(self.returns,self.mov_avg)
        
    @Helper.timing
    def get_target(self):
        yvar = self.yvar
        y_trgt = pd.DataFrame(data=self.data['Dates'])
        for el in self.trgt_days:
            y_trgt[yvar+'_'+str(el)+'D']=self.data[yvar].pct_change(periods=-el)
        self.y_trgt = y_trgt
        
    @Helper.timing
    def process_all(self):
        yvar = self.yvar
        feat_max_days = np.max(self.feat_days)
        trgt_max_days = np.max(self.trgt_days)
        self.features = self.features[self.features[yvar+'_'+str(feat_max_days)+'D'].notna()]
        self.y_trgt = self.y_trgt[self.y_trgt[yvar+'_'+str(trgt_max_days)+'D'].notna()]
        self.feat_trgt = pd.merge(left=self.features,right=self.y_trgt,how='inner',on='Dates',suffixes=(False,False))

    @Helper.timing
    def regr_models(self):
        df = self.feat_trgt.drop(columns=[self.yvar+'_14D'],axis=1)
        a = setup(df,
                  target=self.yvar+'_21D',
                  ignore_features=['Dates'],
                  session_id=11,
                  silent=True,
                  profile=False,
                  remove_outliers=True,
                  )
        self.regrs_models = compare_models(blacklist=self.to_blacklist,turbo=True)

    @Helper.timing
    def get_best_models(self):
        df = self.regrs_models.data
        df.sort_values(by=['R2'],inplace=True)
        models = df['Model'].values[:self.num_selected_models]
        common = set(self.mpg.keys()).intersection(set(models))
        self.selected_models = [self.mpg[el] for el in common]
        self.best_model = self.selected_models[0]
       
    @Helper.timing
    def bagg_tune_best_model(self):
        self.best_model_tuned = tune_model(self.best_model)
        self.best_model_tuned_bagged = ensemble_model(self.best_model_tuned,method='Bagging')
        
    @Helper.timing
    def stacking_model(self):
        models_tuned = []
        for el in [ mdl for mdl in self.selected_models if self.best_model not in mdl ]:
            models_tuned.append(tune_model(el))
        self.model_stack = create_stacknet(estimator_list=[models_tuned,[self.best_model_tuned_bagged]])

    @Helper.timing
    def save_model(self):
        save_model(model=self.model_stack,model_name='model_saved')

    @Helper.timing
    def predict(self):
        regressor = load_model('model_saved')
        predicted_return = predict_model(regressor,data=self.features)
        predicted_return = predicted_return[['Dates','Label']]
        predicted_return.columns = ['Dates','return_'+self.yvar+'_21D']
        predicted_values = self.data[['Dates',self.yvar]]
        predicted_values = predicted_values.tail(len(predicted_return))
        predicted_values = pd.merge(left=predicted_values,right=predicted_return,on=['Dates'],how='inner')
        predicted_values[self.yvar+'_T+21D']=(predicted_values[self.yvar]*
                                              (1+predicted_values['return_'+self.yvar+'_21D'])).round(decimals=2)
        predicted_values['Dates_T+21D'] = predicted_values['Dates']+timedelta(days=21)
        self.predicted_values = predicted_values

        
        
