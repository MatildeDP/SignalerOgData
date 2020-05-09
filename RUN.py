# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:47:11 2020

@author: Værksted Matilde
"""


import numpy as np
from sklearn.model_selection import KFold as Kfold
import matplotlib.pyplot as plt
import string

from text_classifier import TextClassifier
from Baseline import CreateD, bayes, RUN_Baseline
from Process import Preprocess


#%% Import Data: 

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

def Cross(K, Data, labels, Drange,num_epochs):
    """
    Input: 
        K: Number of fold
        Data: list (or other structure) with strings (i.e. words)
        labels: list (or other structure) with labels of corresponding element i Data
    """
    #Initialize arrays and lists for accuracies
    Baseline_AllTrainAccu, Baseline_AllTestAccu = [], []
    Fast_train_accu = np.empty([len(Drange), K])
    Fast_test_accu = np.empty([len(Drange), K])
    
    #Process data
    #Data = Preprocess(Data)
       
    #set seed
    np.random.seed(0)
    
    #Initialise CV    
    fold_idx = 0
    CV =  Kfold(n_splits=K, shuffle=True)
    for trainidx, testidx in CV.split(Data):
        #hej= trainidx
        #hej1 = testidx
                
        Xtrain = Data[trainidx]
        Xtest = Data[testidx]
        ytrain = labels[trainidx]
        ytest = labels[testidx]
        
        
        #Baseline (GloVe and Bayes):    
        #Compute embeddings
        Xtestbase, ytestbase = CreateD(Xtest, ytest)
        Xtrainbase, ytrainbase = CreateD(Xtrain, ytrain)
        
        Baseline_Trainaccu, Baseline_Testaccu = RUN_Baseline(Xtestbase, ytestbase, Xtrainbase, ytrainbase)
        Baseline_AllTrainAccu.append(Baseline_Trainaccu)
        Baseline_AllTestAccu.append(Baseline_Testaccu)
        
        #FastTex:
        for Didx, dim in enumerate(Drange):
            #train classifier
            classifier = TextClassifier(Data,labels, dim, num_epochs=num_epochs)
                        
            #training accuracy
            fasttex_ypredtrain = []
            for text in Xtrain: 
                fasttex_ypredtrain.append(np.argmax(classifier.predict(text, return_prob=True)))
                TrainAccu =  np.sum(fasttex_ypredtrain == ytrain)/len(ytrain)
                Fast_train_accu[Didx,fold_idx] = TrainAccu
                
            
            #test classifier 
            fasttex_ypred = []
            for text in Xtest: 
                fasttex_ypred.append(np.argmax(classifier.predict(text, return_prob=True)))
                TestAccu = np.sum(fasttex_ypred == ytest)/len(ytest)
                Fast_test_accu[Didx,fold_idx] = TestAccu
        
        fold_idx = fold_idx+1
            
  
    return Baseline_AllTrainAccu, Baseline_AllTestAccu, Fast_train_accu, Fast_test_accu


#%%
K =10
Data = train_texts_news
labels  =train_labels_news
Drange = [2, 3, 4, 5, 6]
num_epochs = 2

AllTrainAccu, AllTestAccu,Fast_train_accu, Fast_test_accu = Cross(K, Data, labels, Drange,num_epochs)
# Prepare the x-axis

plt.plot()

