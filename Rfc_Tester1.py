# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 17:34:25 2019

@author: dbda

test hindustan times

"""
import logging
import sys
sys.path.append('F:\PROJECT\Deploy2')
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.ensemble import RandomForestClassifier
from textpackage import data_preprocess as pp
from textpackage import Train_model as tr
from textpackage import Test_model as ts


modelpath = 'F:/PROJECT/Deploy2/'

logging.info('Started NEW EXECUTION')
exit =False
while exit!=True:
    try: 
        print("\n\n============News Atricles Classifier================")
        print("1. Train Model")
        print("2. Test Model")
        print("3. Exit")
        choice=int(input("============Enter Choice: "))
        if choice==1:
            print("*****SELECT MODEL YOU WANT TO BUILD******")
            print("1. SGD CLASSIFIER")
            print("2. RANDOM FOREST")
            print("3. KNN")
            print("4. MULTINOMIAL")
            print("5. VOTING CLASSIFIER")
            print("6. STACK ENSEMBLING")
            print("7. DEEP LEARNIG")
            choiceModel=int(input("============Enter Choice: "))
            
                   
            if choiceModel not in [1,2,3,4,5,6,7]:
                print("\n**********Sorry.... Incorrect Choice ***********")
                continue
            else:
                filepath = pp.select_file()####file path cancel ValueError: Expected object or value
                if filepath != '':
                    data = pp.get_data(filepath)
                    if data == False:
                        print("File doesnot have proper columns ['article', 'title','category']")
                        continue
                    else:
                        X,y = data
                        tr.training(X,y,choiceModel, modelpath)
                else:
                    continue
                #X,y = pp.get_data("file:///F:/PROJECT/Final Models/all_source_title_5080.json")
                
                #tr.training(X,y,choiceModel, modelpath)
            
        elif choice==2:
            print("*****SELECT MODEL YOU WANT TO TEST WITH******")
            print("1. SGD CLASSIFIER")
            print("2. RANDOM FOREST")
            print("3. KNN")
            print("4. MULTINOMIAL")
            print("5. VOTING CLASSIFIER")
            print("6. STACK ENSEMBLING")
            print("7. DEEP LEARNIG")
            choiceModel=int(input("============Enter Choice: "))
            if choiceModel not in [1,2,3,4,5,6,7]:
                print("\n**********Sorry.... Incorrect Choice ***********")
                continue
            else:
                filepath = pp.select_file()####file path cancel ValueError: Expected object or value
                if filepath != '':
                    data = pp.get_data(filepath)
                    if data == False:
                        print("File doesnot have proper columns ['article', 'title','category']")
                        continue
                    else:
                        X,y = data
                        ts.testing(X,y,choiceModel, modelpath)
                else:
                    continue
                
        elif choice==3:
            exit=True
            logging.info('Finished EXECUTION')
            print("\n********** Thank You ....Exiting ***********")
        else:
            print("\n********** Incorrect Choice ***********")
    except Exception as e:
        print("* ",e)
 







