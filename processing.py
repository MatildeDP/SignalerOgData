
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:30:00 2020
@author: Værksted Matilde
"""
import string 
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def Preprocess(TEXT): 
    
    """
    Fjerner tegn fra texts. 
    Hvis text kun består af tegn, og derved bliver en empty string
    fjernes label for tilhørende text 
    """
    
    for i in range(len(TEXT)): 
        # split into words
        tokens = word_tokenize(TEXT[i])
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        # stemming
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in words]
        # join as string again
        TEXT[i]=" ".join(stemmed)
        
        #print(TEXT[i])
        #TEXT[i] = TEXT[i].translate(TEXT[i].maketrans("","",string.punctuation))
        #TEXT[i] = [word for word in tokens if TEXT[i].isalpha()]
    return TEXT


