# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:30:00 2020

@author: Værksted Matilde
"""
import string 
def Preprocess(TEXT): 
    
    """
    Fjerner tegn fra texts. 
    Hvis text kun består af tegn, og derved bliver en empty string
    fjernes label for tilhørende text 
    """
    
    for i in range(len(TEXT)): 
        TEXT[i] = TEXT[i].translate(TEXT[i].maketrans("","",string.punctuation))
    
    return TEXT

