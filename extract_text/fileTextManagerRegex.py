#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textScrapper import TextScrapper
import PyPDF2
import pdfreader
from pdfreader import SimplePDFViewer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize



# In[ ]:


# exemple path file ../data/sncf_vinci_ifg_62/00086570-002601E-B007-20220323-R-VEN.pdf
class fileTextManager():

        
    #def cleanText(self, text):
        # Supprimer la ponctuation
        #text = text.translate(str.maketrans('', '', string.punctuation + '0123456789'))
        
        # Mettre en minuscule
        #text = text.lower()
        
        # Supprimer les stopwords
        #stop_words = set(stopwords.words('french'))
        #words = word_tokenize(text)
        #words = [word for word in words if word not in stop_words]
        
        # Normaliser le texte
        #lemmatizer = WordNetLemmatizer()
        #words = [lemmatizer.lemmatize(word) for word in words]
        
        # Rejoindre les mots en une chaîne de caractères
        #text = ' '.join(words)
        
        #return text
        
    def cleanText(self, text):
        # Supprimer les sauts de ligne, les espaces et les caractères spéciaux
        
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\sàâäéèêëîïôöùûüÿçÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ]', '', text)
        # Supprimer les espaces en trop
        text = ' '.join(text.split()).lower()
        return text

    def getExtension(self, file):
        filename, file_extension = os.path.splitext(file)
        return file_extension
    
    
    def getFileText(self, file):
        textScrap = TextScrapper()
        typeDocumentNumber=[]
        match file["mimeType"]:
            case "application/pdf":
                """
                reader = PyPDF2.PdfFileReader(f"/Volumes/DD_Thibault/sncf_DB/fullDoc/{file['KidFile']}", strict=False)
                if reader.is_encrypted:
                    fullText = textScrap.getTextCryptedPdf(f"/Volumes/DD_Thibault/sncf_DB/fullDoc/{file['KidFile']}")
                    fullText = self.cleanText(fullText)
                else:
                    
                """
                print(file['KidFile'])
                print(file['KidDocumentClass'])
                print(file['database'])
                print("____________________________________")
                
                
                
                fullText = textScrap.getTextPdf(f"/Volumes/DD_Thibault/sncf_DB/fullDoc/{file['KidFile']}")
                fullText = self.cleanText(fullText)
                
                #print (f"{repr(fullText)} file print")
                
               
                return fullText
            case 'application/vnd.ms-excel':
                
                fullText = textScrap.getTextExcel(f"/Volumes/DD_Thibault/sncf_DB/fullDoc/{file['KidFile']}")
                #fullText = fullText[0]
                return fullText

            case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                
                fullText = textScrap.getTextWord(f"/Volumes/DD_Thibault/sncf_DB/fullDoc/{file['KidFile']}")
                #fullText = fullText[0]
                fullText = self.cleanText(fullText)
                return fullText
            
            #case 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                #fullText = textScrap.getTextWord(f"/Volumes/DD_Thibault/sncf_DB/doc/engie/{file['KidFile']}")
                #return fullText
            #case 'image/jpeg':
                #return 2

            case _:
                return 'document type not supported'
                #raise ValueError('document type not supported')   # 0 is the default case if x is not found
    

