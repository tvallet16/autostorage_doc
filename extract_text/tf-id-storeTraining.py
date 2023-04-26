#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import nltk
import PyPDF2

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from textScrapper import TextScrapper
from fileTextManagerRegex import fileTextManager
from PyPDF2 import PdfReader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
warnings.filterwarnings('ignore')


# In[2]:


base="sncf_veolia_ifg_21"
csvPath = f"/Volumes/DD_Thibault/sncf_DB/storeTraining/fusion.csv"
if not os.path.exists("models"):
    os.makedirs("models")

scrapper = TextScrapper()
csv = scrapper.getTextCsv(csvPath)
fileManager = fileTextManager()
#print(len(csv))
#print(csv.iloc[4])
orderedCorpus = {}
dictType={}
for row in tqdm((csv.iloc)):
    if row["mimeType"] == "application/pdf" or row["mimeType"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        if row['KidDocumentClass'] in dictType:
            if row['database'] in dictType[row['KidDocumentClass']]:
                if len(dictType[row['KidDocumentClass']][row['database']]) <200:
                    contentFile = fileManager.getFileText(row)
                    dictType[row['KidDocumentClass']][row['database']].append(contentFile)
            else:
                contentFile = fileManager.getFileText(row)
                dictType[row['KidDocumentClass']][row['database']] = []
                dictType[row['KidDocumentClass']][row['database']].append(contentFile)
        else:
            dictType[row['KidDocumentClass']]={}
            
            if row['database'] in dictType[row['KidDocumentClass']]:
                contentFile = fileManager.getFileText(row)
                dictType[row['KidDocumentClass']][row['database']].append(contentFile)
            else:

                contentFile = fileManager.getFileText(row)
                dictType[row['KidDocumentClass']][row['database']] = []
                dictType[row['KidDocumentClass']][row['database']].append(contentFile)

import json
with open('data200shufleRegClean.json', 'w', encoding='utf-8') as f:
    json.dump(dictType, f, ensure_ascii=False, indent=4)




# In[7]:


data = {}
for key in dictType:
    
    for bases in dictType[key]:
       
        
        for file in bases:

            dictType[key][bases] = list(filter(None, dictType[key][bases]))
        print("_______________")
        if len(dictType[key][bases]) >= 10:
            if key in data:
                if bases in data[key]:
                    data[key][bases]["data"], data[key][bases]["test"]= train_test_split(dictType[key][bases],test_size=0.03)
                    
                else:
                    data[key][bases] = {"data":[], "test":[]}
                    data[key][bases]["data"], data[key][bases]["test"]= train_test_split(dictType[key][bases],test_size=0.03)
                    
            else :
                data[key] = {}
                if bases in data[key]:
                    data[key][bases]["data"], data[key][bases]["test"]= train_test_split(dictType[key][bases],test_size=0.03)
                
                else:
                    data[key][bases] = {"data":[], "test":[]}
                    data[key][bases]["data"], data[key][bases]["test"]= train_test_split(dictType[key][bases],test_size=0.03)


# In[8]:


def trainModel():
    for types in data:
        for bases in data[types]:
            vecsData = {}
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(data[types][bases]["data"])
            #mean = np.mean(vectors.toarray(), axis=0)
            vecsData["vectors"] = vectors
            vecsData["vectorizer"] = vectorizer
            with open(os.path.join("models", f'{types}:{bases}.pkl'), 'wb') as f:
                pickle.dump(vecsData, f)

    
    return vecsData
        
    


# In[9]:


trainModel()


# In[20]:


def compareDocSave(doc):
    tamp = -2
    similarities = []
    files = os.listdir("models")
    sim=[]
    for file in files:
        types = file
        
        if os.path.isfile(f"models/{file}"):
            with open(os.path.join("models", file), 'rb') as f:
                vecsData = pickle.load(f)
            print(vecsData)
            for vec in vecsData["vectors"]:
                similarity = np.dot(vecsData["vectorizer"].transform([doc]).toarray()[0], vec.toarray()[0]) / (np.linalg.norm(vecsData["vectorizer"].transform([doc]).toarray()[0]) * np.linalg.norm(vec.toarray()[0]))

                if tamp < similarity:
                    tamp = similarity
                    docType = os.path.splitext(types)[0]+''
                    sim = [docType, similarity]
    if sim == []:
        return "error"
                    
    return sim


# In[18]:


lenth = 0
lenthTest=0
for types in data:
    for bases in data[types]:
        lenth = lenth + len(data[types][bases]["data"])
        lenthTest = lenthTest + len(data[types][bases]["test"])
    
print(lenth)
print(lenthTest)


# In[21]:


fail= 0
succes=0
failList= []
err=0
for types in tqdm(data):
    typeSucces = 0
    typeFail = 0
    for bases in data[types]:
        for file in data[types][bases]["test"]:

            simFile = compareDocSave(file)
            print(simFile)

            if simFile != "error":
                if simFile[0].split(":", 1)[0] == types:
                    succes +=1
                    typeSucces += 1
                else:
                    typeFail += 1
                    fail+=1
                    failList.append(simFile[0])
            else:
                err+=1
    print(f"for {types} :")
    print(f"err:  {err}")        
    print(f"fail:  {typeFail}")
    print(f"succes:  {typeSucces}")
    print(f"{100-(typeFail*100/(typeFail+typeSucces))} % reliable")

print(f"err:  {err}")        
print(f"fail:  {fail}")
print(f"succes:  {succes}")
print(f"{100-(fail*100/(fail+succes))} % reliable")


# In[13]:


for types in data:
    print(types)


# In[3]:


import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textScrapper import TextScrapper
from PyPDF2 import PdfReader
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

def clean_text(text):
    # Supprimer les sauts de ligne, les espaces et les caractères spéciaux
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Supprimer les espaces en trop
    text = ' '.join(text.split())

    return text

textScrap = TextScrapper()
fullText = textScrap.getTextPdf(f"K_NODE;0001395Q6KBE0501")
print(fullText)
print("____________________________________________")
fullText =clean_text(fullText)

print(fullText)


# In[5]:





# In[ ]:





# In[ ]:




