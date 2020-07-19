# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:42:51 2020

@author: Devansh Dhrafani
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.random import sample_without_replacement
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,accuracy_score
    )

def model1(trdata,tract,tsdata):
    model = LogisticRegression()
    model.fit(trdata,tract)
    pred= model.predict(tsdata)
    return pred

def model2(trdata,tract,tsdata):
    model = DecisionTreeClassifier()
    model.fit(trdata,tract)
    pred= model.predict(tsdata)
    return pred

accuracy=np.zeros((56,6))
fmeasure=np.zeros((56,6))
accuracyBP=np.zeros((112,3))
fmeasureBP=np.zeros((112,3))
accuracy2BP=np.zeros((168,2))
fmeasure2BP=np.zeros((168,2))

for project in range(0,56):
    acc=np.zeros((1,6))
    fmea=np.zeros((1,6))
    
    fname='E:/College/__Ongoing/3-3 Summer Term/Machine Learning/2. Data Clustering/Machine Learning -L6/'+str(project+1)+'.csv'
    df=pd.read_csv(fname,header=None,encoding = 'unicode_escape')
    df[20] =  [1 if b>=1 else 0 for b in df[20]]
    
    print('project no: '+str(project+1))
    
    x=df.values
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled)
    
    # preparing data for up/down sampling
    df_majority = data[data[20]==0]
    df_minority = data[data[20]==1]
    label_counts = data[20].value_counts()
    
    # Random Sampling
    y_all = data.pop(20)
    X_all = data
    trdata, tsdata, tract, tsact = train_test_split(X_all.values,y_all.values,test_size=0.2,train_size=0.6)
    
    pred=model1(trdata,tract,tsdata)
    acc[0,0]=accuracy_score(tsact,pred)
    fmea[0,0]=f1_score(tsact,pred,zero_division=0)
    pred=model2(trdata,tract,tsdata)
    acc[0,1]=accuracy_score(tsact,pred)
    fmea[0,1]=f1_score(tsact,pred,zero_division=0)
    
    # Up Sampling
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=label_counts.max() )   # to match majority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    y_all = df_upsampled.pop(20)
    X_all = df_upsampled
    trdata, tsdata, tract, tsact = train_test_split(X_all.values,y_all.values)
    
    pred=model1(trdata,tract,tsdata)
    acc[0,2]=accuracy_score(tsact,pred)
    fmea[0,2]=f1_score(tsact,pred,zero_division=0)
    pred=model2(trdata,tract,tsdata)
    acc[0,3]=accuracy_score(tsact,pred)
    fmea[0,3]=f1_score(tsact,pred,zero_division=0)
        
    # Down Sampling
    df_majority_downsampled = resample(df_majority, 
                                     replace=False,     # sample with replacement
                                     n_samples=label_counts.min() )   # to match majority class
    
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    
    y_all = df_downsampled.pop(20)
    X_all = df_downsampled
    trdata, tsdata, tract, tsact = train_test_split(X_all.values,y_all.values)
    
    pred=model1(trdata,tract,tsdata)
    acc[0,4]=accuracy_score(tsact,pred)
    fmea[0,4]=f1_score(tsact,pred,zero_division=0)
    pred=model2(trdata,tract,tsdata)
    acc[0,5]=accuracy_score(tsact,pred)
    fmea[0,5]=f1_score(tsact,pred,zero_division=0)
    
    accuracy[project,:] = acc[0,:]
    fmeasure[project,:] = fmea[0,:]
    
    acccla=np.zeros((2,3))
    fmeacla=np.zeros((2,3))
    for i in range(0,2):
        acccla[(i):(i+1),:]=acc[:,3*(i):3*(i+1)]
        fmeacla[(i):(i+1),:]=fmea[:,3*(i):3*(i+1)]
    
    accuracyBP[2*project:(2*project+2), :]=acccla
    fmeasureBP[2*project:(2*project+2), :]=fmeacla
    
    accds=np.zeros((3,2))
    fmeads=np.zeros((3,2))
    for i in range(0,3):
        for j in range(0,2):
            accds[i,j]=acc[0,3*j+i]
            fmeads[i,j]=fmea[0,3*j+i]
                
    accuracy2BP[3*project:(3*project+3), :]=accds
    fmeasure2BP[3*project:(3*project+3), :]=fmeads

pyplot.boxplot(accuracyBP)
pyplot.grid(True)
fna='C:/Users/Devansh Dhrafani/Desktop/ML Assignment/accvalue.png'
pyplot.savefig(fna)
pyplot.close()
pyplot.boxplot(fmeasureBP)
pyplot.grid(True)
fna='C:/Users/Devansh Dhrafani/Desktop/ML Assignment/fmeavalue.png'
pyplot.savefig(fna)
pyplot.close()

pyplot.boxplot(accuracy2BP)
pyplot.grid(True)
fna='C:/Users/Devansh Dhrafani/Desktop/ML Assignment/acc2value.png'
pyplot.savefig(fna)
pyplot.close()
pyplot.boxplot(fmeasure2BP)
pyplot.grid(True)
fna='C:/Users/Devansh Dhrafani/Desktop/ML Assignment/fmea2value.png'
pyplot.savefig(fna)
pyplot.close()
