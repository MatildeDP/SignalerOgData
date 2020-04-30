# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:40:22 2020

@author: Værksted Matilde
"""

import numpy as np
import torch
from text_classifier import TextClassifier
import string

##Baseline 
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
train_texts = spam_data['train_texts']
test_texts = spam_data['test_texts']
train_labels = spam_data['train_labels']
test_labels = spam_data['test_labels']
spamlabels = spam_data['labels']