# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:44:29 2020

@author: Værksted Matilde
"""

from text_classifier import TextClassifier
import numpy as np
import torch
import torchtext
import tqdm

##Baseline aka GloVe

filename = "./Data/glove.6B.50d.txt"


dictionary = {}
with open(filename,'r', encoding='utf-8') as file:
    for line in file:
        elements = line.split();
        word = elements[0];
        vector = np.asarray(elements[1:],"float32") #embedding of word line
        dictionary[word] = vector; # collect mean od this word
        
        
        
#Fasttex
