# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:04:13 2020

@author: Bruger
"""


from Dataloader import *
#Create D: 
def CreateD(TEXT):
    """ 
    Indput: 
    TEXT: Array with sentences
    """
    Dtotal = []
    i = 0
    
    for text in TEXT:
        print(i)#Loop through each sentence
        D = []
        text = text.translate(text.maketrans("","",string.punctuation))
        print(text)
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
            #Dtotal.append(D)
    return Dtotal

              
D_train = CreateD(train_texts)               
D_test = CreateD(test_texts) 


