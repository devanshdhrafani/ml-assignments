# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 11:47:08 2020

@author: Devansh Dhrafani
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,accuracy_score
    )

def model1(trdata,tract,tsdata):
    model = BernoulliNB()
    model.fit(trdata,tract)
    pred= model.predict(tsdata)
    return pred
def model2(trdata,tract,tsdata):
    model = MultinomialNB()
    model.fit(trdata,tract)
    pred= model.predict(tsdata)
    return pred

def model3(trdata,tract,tsdata):
    model = GaussianNB()
    model.fit(trdata,tract)
    pred= model.predict(tsdata)
    return pred

accuracy=np.zeros((56,12))             #overall data
fmeasure=np.zeros((56,12))             #overall data
accuracyBP=np.zeros((224,3))            #model wise accuracy
fmeasureBP=np.zeros((224,3))            #model wise fmeasure
accuracy2BP=np.zeros((168,4))           #feature selection technique wise accuracy
fmeasure2BP=np.zeros((168,4))           #feature selection technique wise fmeasure
NumFeatSelect = np.zeros((56,3))

for project in range(0,56):
    fname='E:/College/__Ongoing/3-3 Summer Term/Machine Learning/2. Data Clustering/Machine Learning -L6/'+str(project+1)+'.csv'
    data=pd.read_csv(fname,header=None,encoding = 'unicode_escape')
    
    print('project no: '+str(project+1))
    x=data.iloc[:, :20]
    datan=x.values
    scaler = MinMaxScaler()
    datan=scaler.fit_transform(datan)
    
    outclass=data.iloc[:, -1].values
    index1=np.where(outclass>1)
    outclass[index1[0]]=1
    outclass.resize(outclass.shape[0],1)
    #print(x)
    #print(outclass)
    
    X=datan
    y=outclass
    
    # GINI Index Decision Tree
    clf = DecisionTreeClassifier(criterion='gini', max_features="log2")
    clf.fit(X, y)
    
    # Info Gain Decision Tree
    clf2 = DecisionTreeClassifier(criterion='entropy', max_features="log2")
    clf2.fit(X, y)
    
    #graph = Source(export_graphviz(clf, out_file=None))
    #graph.format = 'png'
    #saveFile='C:/Users/Devansh Dhrafani/Desktop/ML Assignment/Trees/'+str(project+1)+'_GINI.png'
    #graph.render(saveFile, view=False);
    
    #graph = Source(export_graphviz(clf2, out_file=None))
    #graph.format = 'png'
    #saveFile='C:/Users/Devansh Dhrafani/Desktop/ML Assignment/Trees/'+str(project+1)+'_INFO_GAIN.png'
    #graph.render(saveFile, view=False);
    
    # Find imp features by GINI Index Tree
    featImp=clf.feature_importances_
    impFeatureIndex=np.where(featImp>0)
    datan1=datan[:,impFeatureIndex[0]]
    NumFeatSelect[project,0]=datan1.shape[1]
    
    # Find imp features by Info Gain Tree
    featImp2=clf2.feature_importances_
    impFeatureIndex2=np.where(featImp2>0)
    datan2=datan[:,impFeatureIndex2[0]]
    NumFeatSelect[project,1]=datan2.shape[1]

    trdata, tsdata, tract, tsact = train_test_split(datan,outclass)
    
    # PCA to find imp features
    pca = PCA()
    datan = pca.fit(datan)
    CumExplainedVariance = np.cumsum(pca.explained_variance_ratio_)
    SigFeaturesPCA=np.where(CumExplainedVariance>0.99)
    numComponents= SigFeaturesPCA[0][0] +1 
    NumFeatSelect[project,2]=numComponents
    
    # Get data ready of significant features using PCA
    pca = PCA(n_components=numComponents)
    X_train=trdata
    X_test=tsdata
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.fit_transform(X_test)
    
    
    #pyplot.plot(np.cumsum(pca.explained_variance_ratio_))
    #pyplot.xlabel('number of components')
    #pyplot.ylabel('cumulative explained variance');
    #print(pca.explained_variance_)
    
    acc=np.zeros((1,12))
    fmea=np.zeros((1,12))
    
    cou=0
    
    # All features
    pred=model1(trdata,tract.ravel(),tsdata)
    acc[cou,0]=accuracy_score(tsact,pred)
    fmea[cou,0]=f1_score(tsact,pred,zero_division=0)
    pred=model2(trdata,tract.ravel(),tsdata)
    acc[cou,1]=accuracy_score(tsact,pred)
    fmea[cou,1]=f1_score(tsact,pred,zero_division=0)
    pred=model3(trdata,tract.ravel(),tsdata)
    acc[cou,2]=accuracy_score(tsact,pred)
    fmea[cou,2]=f1_score(tsact,pred,zero_division=0)
    
    # Get data ready of significant features using GINI
    trdata, tsdata, tract, tsact = train_test_split(datan1,outclass)
    
    # GINI
    pred=model1(trdata,tract.ravel(),tsdata)
    acc[cou,3]=accuracy_score(tsact,pred)
    fmea[cou,3]=f1_score(tsact,pred,zero_division=0)
    pred=model2(trdata,tract.ravel(),tsdata)
    acc[cou,4]=accuracy_score(tsact,pred)
    fmea[cou,4]=f1_score(tsact,pred,zero_division=0)
    pred=model3(trdata,tract.ravel(),tsdata)
    acc[cou,5]=accuracy_score(tsact,pred)
    fmea[cou,5]=f1_score(tsact,pred,zero_division=0)
    
    # Get data ready of significant features using Info Gain
    trdata, tsdata, tract, tsact = train_test_split(datan2,outclass)
    
    # Info Gain
    pred=model1(trdata,tract.ravel(),tsdata)
    acc[cou,6]=accuracy_score(tsact,pred)
    fmea[cou,6]=f1_score(tsact,pred,zero_division=0)
    pred=model2(trdata,tract.ravel(),tsdata)
    acc[cou,7]=accuracy_score(tsact,pred)
    fmea[cou,7]=f1_score(tsact,pred,zero_division=0)
    pred=model3(trdata,tract.ravel(),tsdata)
    acc[cou,8]=accuracy_score(tsact,pred)
    fmea[cou,8]=f1_score(tsact,pred,zero_division=0)
    
    
    # PCA
    pred=model1(X_train,tract.ravel(),X_test)
    acc[cou,9]=accuracy_score(tsact,pred)
    fmea[cou,9]=f1_score(tsact,pred,zero_division=0)
    pred=model2(X_train,tract.ravel(),X_test)
    acc[cou,10]=accuracy_score(tsact,pred)
    fmea[cou,10]=f1_score(tsact,pred,zero_division=0)
    pred=model3(X_train,tract.ravel(),X_test)
    acc[cou,11]=accuracy_score(tsact,pred)
    fmea[cou,11]=f1_score(tsact,pred,zero_division=0)
    
    cou=cou+1
    
    accuracy[project,:] = acc[0,:]
    fmeasure[project,:] = fmea[0,:]
    
    acccla=np.zeros((4,3))
    fmeacla=np.zeros((4,3))
    for i in range(0,4):
        acccla[(i):(i+1),:]=acc[:,3*(i):3*(i+1)]
        fmeacla[(i):(i+1),:]=fmea[:,3*(i):3*(i+1)]
    
    accuracyBP[4*project:(4*project+4), :]=acccla
    fmeasureBP[4*project:(4*project+4), :]=fmeacla
    
    accds=np.zeros((3,4))
    fmeads=np.zeros((3,4))
    for i in range(0,3):
        for j in range(0,4):
            accds[i,j]=acc[0,3*j+i]
            fmeads[i,j]=fmea[0,3*j+i]
                
    accuracy2BP[3*project:(3*project+3), :]=accds
    fmeasure2BP[3*project:(3*project+3), :]=fmeads
    
pyplot.boxplot(accuracyBP)
pyplot.grid(True)
fna='C:/Users/Devansh Dhrafani/Desktop/ML Assignment/Part2/accvalue.png'
pyplot.savefig(fna)
pyplot.close()
pyplot.boxplot(fmeasureBP)
pyplot.grid(True)
fna='C:/Users/Devansh Dhrafani/Desktop/ML Assignment/Part2/fmeavalue.png'
pyplot.savefig(fna)
pyplot.close()

pyplot.boxplot(accuracy2BP)
pyplot.grid(True)
fna='C:/Users/Devansh Dhrafani/Desktop/ML Assignment/Part2/acc2value.png'
pyplot.savefig(fna)
pyplot.close()
pyplot.boxplot(fmeasure2BP)
pyplot.grid(True)
fna='C:/Users/Devansh Dhrafani/Desktop/ML Assignment/Part2/fmea2value.png'
pyplot.savefig(fna)
pyplot.close()