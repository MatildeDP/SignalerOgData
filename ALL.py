# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:10:32 2020

@author: Værksted Matilde
"""

import numpy as np

from text_classifier import TextClassifier
import numpy as np

##Baseline aka GloVe
filename = "./Data/glove.6B.50d.txt"


dictionary = {}
with open(filename,'r', encoding='utf-8') as file:
    for line in file:
        elements = line.split();
        word = elements[0];
        vector = np.asarray(elements[1:],"float32") #embedding of word line
        dictionary[word] = vector; # collect mean od this word
     
#Import SPAM Data
filename_SPAM = './Data/spam_data.npz'
spam_data = np.load(filename_SPAM, allow_pickle=True)
train_texts_spam = spam_data['train_texts']
test_texts_spam = spam_data['test_texts']
train_labels_spam = spam_data['train_labels']
test_labels_spam = spam_data['test_labels']
spamlabels = spam_data['labels']


#Import NEWS Data
news_data = np.load('./Data/news_data.npz', allow_pickle=True)
train_texts_news = news_data['train_texts']
test_texts_news = news_data['test_texts']
train_labels_news = news_data['train_labels']
test_labels_news = news_data['test_labels']
ag_news_labels = news_data['ag_news_label']



#%%


from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold as Kfold
import numpy as np
import matplotlib.pyplot as plt
import string
from Models import *


#Create D: 
def CreateD(TEXT, label):
    """ 
    Indput: 
    TEXT: Array with sentences
    """
    Dtotal = []
    delete = []
    i = 0
    
    for i, text in enumerate(TEXT):
        #print(i)#Loop through each sentence
        D = [] 
        
        text = text.translate(text.maketrans("","",string.punctuation))

        for word in text.split(): #Loop through each word
            if word in dictionary: 
                D.append(dictionary[word])
            else: 
                pass
            
        D = np.asarray(D).T
        if len(D)!=0:     
            m = np.mean(D,axis=1)
            Dtotal.append(m)
            i =i+1
        else: 
            delete.append(i)
            #Dtotal.append(D)
    Dtotal = np.asarray(Dtotal).T
    print(delete)
    label = np.delete(label,delete)
    
    print("D has been computed")
    return Dtotal.T, label



def bayes(NewData, mean, cov, prior):   
    """
    Bayes classifier using multivariate normal distribtution
    """
    
    like, evidence, prob = [], [], []
    for i in range(len(mean)): 
        like.append(multivariate_normal.pdf(NewData,mean[i],cov[i]))
        
    evidence = sum(like[j]*prior[j] for j in range(len(mean)))
    #loglike0*prior[0] + loglike1*prior[1]
    
    for j in range(len(mean)): 
        prob.append(like[j]*prior[j]/evidence)
    
    return prob, np.argmax(prob)



def Baseline(Xtest, ytest, Xtrain, ytrain):
    """
    Trains and test Bayes on input
    Returns test and training accuracy
    """
    mean, cov, Train_pred, Test_pred = [], [], [], []
    
    
    if (Xtest.shape[0] < Xtest.shape[1]): 
        print("Transpose Xtest in order to get the correct output")
        return 
        
    elif ((Xtrain.shape[0] < Xtrain.shape[1])): 
        print("Transpose Xtrain in order to get the correct output")
        return
    #compute prior
    uni, freqtrain = np.unique(ytrain, return_counts=True)
    prior = (freqtrain)/len(ytrain)
    
    #number of classes and observations: 
    M = len(freqtrain)
    Ntrain = Xtrain.shape[0]
    Ntest = Xtest.shape[0]
    
    #compute mean and cov: 
    for i in range(M): 
        X = Xtrain[ytrain == i]
        mean.append(np.mean(X,0))
        cov.append(np.cov(X.T))
    

    for i in range(Ntrain): 
        Train_pred.append(bayes(Xtrain[i,:], mean, cov, prior)[1]) 
        print("Prediction, train observation %s" %i )
    print("Classes of all training observations has been predicted")
    

    for i in range(Ntest): 
        Test_pred.append(bayes(Xtest[i,:], mean, cov, prior)[1])
        print("Prediction, test observation %s" %i )
        
    print("Classes of all test observations has been predicted")
        
    ##training acc: 
    Train_pred = np.asarray(Train_pred)
    TrainAccu = (np.sum(Train_pred == ytrain))/Ntrain
    
    ##test accu
    Test_pred = np.asarray(Test_pred)
    TestAccu = np.sum(Test_pred == ytest)/Ntest
    
    return TrainAccu, TestAccu
    

        
#%%
#Run Baseline with Spam data
Dtrain, trainlabel = CreateD(train_texts_spam, train_labels_spam)
Dtest, testlabel = CreateD(test_texts_spam, test_labels_spam)  
Trainaccuspam, Testaccuspam = Baseline(Dtest, testlabel, Dtrain, trainlabel)


Dtrain, trainlabel = CreateD(train_texts_news, train_labels_news)
Dtest, testlabel = CreateD(test_texts_news, test_labels_news)  
Trainaccunews, Testaccunews = Baseline(Dtest, testlabel, Dtrain, trainlabel)
