
import logging
# Change your own path instead of F:/PROJECT/Deploy2
import sys
sys.path.append('F:/PROJECT/Deploy2')
import pandas as pd
from textpackage import data_preprocess as pp
from textpackage import Train_model as tr
from textpackage import Test_model as ts


modelpath = 'F:/PROJECT/Deploy2/'

logging.info('*** Started new Execution of the Application')
exit = False
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
                filepath = pp.select_file()
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
            logging.info('*** Closing the Application')
            print("\n********** Thank You .... Exiting ***********")
        else:
            print("\n********** Incorrect Choice ***********")
    except Exception as e:
        print("* ",e)
 







