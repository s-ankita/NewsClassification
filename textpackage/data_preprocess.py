# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 20:29:38 2019

@author: dbda
"""
import re
import copy
import pandas as pd
from textblob import Word
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
import tkinter
from tkinter import filedialog

def select_file():

    root=tkinter.Tk()
    #root.withdraw()
    #root.attributes("-topmost", True)
    #root.focus_force()
    root.lift()
    root.attributes('-topmost',True)
    root.after_idle(root.attributes,'-topmost',False)
    #root.lift()
     # Close the root window
    filetypes = [('JSON files', '.json')]
    in_path = filedialog.askopenfilename(initialdir=r'F:\PROJECT\Deploy2\Datasets', filetypes = filetypes)
    #in_path = filedialog.askopenfilename(initialdir=r'F:\PROJECT\Deploy2\Datasets')
    root.withdraw()
    return in_path
    

def file_save():
    f = filedialog.asksaveasfile(mode='w', defaultextension=".joblib")
    print(f.name)
    
    if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
        filename = 'F:\PROJECT\Deploy2\Models\default.joblib'
    else:
        filename = f.name
    #text2save = str(text.get(1.0, END)) # starts from `1.0`, not `0.0`
    #f.write(text2save)
    f.close() # `()` was missing.
    return filename


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



def preprocess(x):
    print("\n\n============Preprocessing Data================")
    x=x.tolist()
    #ps = PorterStemmer()
    for index,value in enumerate(x):
        #print("processing data:",index)
        x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])
        #x[index] = ' '.join([ps.stem(word) for word in clean_str(value).split()])
        #y=pd.Series(x)
    return pd.Series(x)

def get_data(filepath):
    #data = pd.read_json("file:///F:/PROJECT/Final Models/all_source_title_5080.json")
    #data = pd.read_excel("file:///F:/PROJECT/Deploy2/Datasets/HINDUSTAN_TIMES_FINAL_DATA.xlsx")
    data = pd.read_json(filepath)
    data=data.dropna()
    
    if True == all(elem in data.columns  for elem in ['article', 'category', 'title']):
        X = data['title'] + " " + data['article']
        y = data['category']
        X = preprocess(X)
        lbcode = LabelEncoder()
        y = lbcode.fit_transform(y)
        return(X,y)
    else:
        return False
        
    
    
def add_new_category(X,y):
    data = pd.read_json("file:///F:/PROJECT/Final Models/all_source_title_5080.json")
    X_new_cat = data['title'] + " " + data['article']
    y_new_cat = data['category']
    y_new_cat[:] = 5
    X_new_cat = preprocess(X_new_cat)
    return(X,y)


if __name__ == "__main__":
    pass
#    data = pd.read_json("file:///F:/PROJECT/Final Models/all_source_title_5080.json")
#    data_text=pd.DataFrame(data)
#    x = (data['title'] +" "+ data['article']).tolist()
#    x_old=copy.deepcopy(x)  
#    x = preprocess(data['title'] +" "+ data['article'])
    
#    import py_compile
#    py_compile.compile(r'F:\PROJECT\Final Models\data_preprocess.py')

    
    
    
    

    
    
    
    
    