import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
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

accuracy=np.zeros((56,6))
fmeasure=np.zeros((56,6))
accuracyBP=np.zeros((560,3))
fmeasureBP=np.zeros((560,3))
accuracy2BP=np.zeros((840,2))
fmeasure2BP=np.zeros((840,2))

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
    #print(x)
    #print(outclass)
    
    in0=np.where(outclass==0)
    in1=np.where(outclass==1)
    p=np.zeros((datan.shape[1]))
    for i in range(0,datan.shape[1]):
        f=datan[:, i]
        f0=f[in0[0]]
        f1=f[in1[0]]
        try:
            w,p[i]=mannwhitneyu(f0,f1)
        except:
            continue
    in1=np.where(p<=0.05)
    datan1=datan[:,in1[0]]
    
    
    kf = KFold(5)
    
    acc=np.zeros((5,6))
    fmea=np.zeros((5,6))
    cou=0
    for train_index, test_index in kf.split(datan):
        trdata=datan[train_index,:]
        tsdata=datan[test_index,:]
        tract=outclass[train_index]
        tsact=outclass[test_index]
        pred=model1(trdata,tract,tsdata)
        acc[cou,0]=accuracy_score(tsact,pred)
        fmea[cou,0]=f1_score(tsact,pred,zero_division=0)
        pred=model2(trdata,tract,tsdata)
        acc[cou,1]=accuracy_score(tsact,pred)
        fmea[cou,1]=f1_score(tsact,pred,zero_division=0)
        pred=model3(trdata,tract,tsdata)
        acc[cou,2]=accuracy_score(tsact,pred)
        fmea[cou,2]=f1_score(tsact,pred,zero_division=0)
    
        trdata=datan1[train_index,:]
        tsdata=datan1[test_index,:]
        pred=model1(trdata,tract,tsdata)
        acc[cou,3]=accuracy_score(tsact,pred)
        fmea[cou,3]=f1_score(tsact,pred,zero_division=0)
        pred=model2(trdata,tract,tsdata)
        acc[cou,4]=accuracy_score(tsact,pred)
        fmea[cou,4]=f1_score(tsact,pred,zero_division=0)
        pred=model3(trdata,tract,tsdata)
        acc[cou,5]=accuracy_score(tsact,pred)
        fmea[cou,5]=f1_score(tsact,pred,zero_division=0)
        cou=cou+1
    
    accuracy[project,:] = acc.mean(axis=0)
    fmeasure[project,:] = fmea.mean(axis=0)
    
    acccla=np.zeros((10,3))
    fmeacla=np.zeros((10,3))
    for i in range(0,2):
        acccla[5*(i):5*(i+1),:]=acc[:,3*(i):3*(i+1)]
        fmeacla[5*(i):5*(i+1),:]=fmea[:,3*(i):3*(i+1)]
    
    accuracyBP[10*project:(10*project+10) ,:]=acccla
    fmeasureBP[10*project:(10*project+10) ,:]=fmeacla

    accds=np.zeros((15,2))
    fmeads=np.zeros((15,2))
    for j in range(0,2):
        for i in range(0,3):
            accds[5*(i):5*(i+1),j]=acc[:,i+3*j]
            fmeads[5*(i):5*(i+1),j]=fmea[:,i+3*j]
        
    accuracy2BP[15*project:(15*project+15) ,:]=accds
    fmeasure2BP[15*project:(15*project+15) ,:]=fmeads
        
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

pval=np.zeros((3,3))
pval1=np.zeros((3,3))
for i in range(0,3):
    for j in range(0,3):
        w,pval[i,j]=mannwhitneyu(fmeasureBP[:,i],fmeasureBP[:,j])
        w,pval1[i,j]=mannwhitneyu(accuracyBP[:,i],accuracyBP[:,j])


