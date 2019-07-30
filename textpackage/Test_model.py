# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 17:34:25 2019

@author: dbda

Functions for Training Model

"""
import logging
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from joblib import load
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score


modelpath = 'F:/PROJECT/Deploy2/'
model_names={1:'SGD',2:'RF',3:'KNN',4:'MNB',5:'VOTING',6:'STACK'}

def load_vocab(modelpath,name):
    print(modelpath + 'vocabularies/'+name+'.pickle')
    pickle_in = open(modelpath + 'vocabularies/'+name+'.pickle',"rb")
    vocab = pickle.load(pickle_in)
    pickle_in.close()
    return vocab


def feature_extraction(X,choiceModel, modelpath):
    print("\n\n============Extracting features================")
    logging.info('Started feature Extraction ')
    vocab = load_vocab(modelpath,model_names.get(choiceModel))
    tfidf_vect = TfidfVectorizer(analyzer='word',min_df=3, token_pattern=r'[a-zA-Z]{2,}', vocabulary=vocab, stop_words = 'english')
    xtest_tfidf =  pd.DataFrame(tfidf_vect.fit_transform(X).todense(),columns=tfidf_vect.get_feature_names())
    logging.info('Finished feature Extraction ')
    return xtest_tfidf

def stack_ensembling_predict(xtest_tfidf,y,choiceModel, modelpath):
    knn = load( modelpath + 'STACK_KNN.joblib') 
    knn_pred_test = knn.predict(xtest_tfidf)
    knn_pred_test = pd.get_dummies(knn_pred_test, drop_first = True)
    rfc = load( modelpath + 'STACK_RFC.joblib')     
    rfc_pred_test = rfc.predict(xtest_tfidf) 
    rfc_pred_test = pd.get_dummies(rfc_pred_test, drop_first = True)
    sgdc = load( modelpath + 'STACK_SGDC.joblib')     
    sgdc_pred_test = sgdc.predict(xtest_tfidf)
    sgdc_pred_test = pd.get_dummies(sgdc_pred_test, drop_first = True)
    lr = load( modelpath + 'STACK_LEVEL2_LR.joblib')     
    X_test_pred = pd.concat([knn_pred_test, rfc_pred_test, sgdc_pred_test ], axis=1)
    X_test_pred.shape
    
    y_pred = lr.predict(xtest_tfidf)
    y_pred_proba = lr.predict_proba(xtest_tfidf)
    return (y_pred,y_pred_proba)


def predict(xtest_tfidf,y,choiceModel, modelpath):
    print("\n\n============Loading model for prediction...================")
    logging.info('Loading Model For Prediction')
    if choiceModel==1:
        model = load( modelpath + 'Models/SGD_CLASSIFIER.joblib')     
    elif choiceModel==2:
        model = load( modelpath + 'Models/RANDOM_FOREST.joblib')
    elif choiceModel==3:
        model = load( modelpath + 'Models/K_NN_CLASSIFIER.joblib')
    elif choiceModel==4:
        model = load( modelpath + 'Models/MULTINOMAIL_NAIVE_BAYES.joblib') 
    elif choiceModel==5:
        model = load( modelpath + 'Models/VOTING_CLASSIFIER.joblib')
    elif choiceModel==6:
        return stack_ensembling_predict(xtest_tfidf,y,choiceModel, modelpath)
    y_pred = model.predict(xtest_tfidf)
    y_pred_proba = model.predict_proba(xtest_tfidf)
    return (y_pred,y_pred_proba)

def predict_dl(texts_test,target_test,choiceModel, modelpath):
    logging.info('Loading Model For Prediction and predicting')
    max_length=300
    tokenizer = load(modelpath + 'Models/DEEP_LEARNING_TOKENIZER.joblib')
    sequences_test=tokenizer.texts_to_sequences(texts_test)
    data_test = pad_sequences(sequences_test, maxlen=max_length)
    #labels_test = to_categorical(np.asarray(target_test))
    
    model = load( modelpath + 'Models/DEEP_LEARNING.joblib')
    print("PREDICTING")
    pred = model.predict(data_test)
    print("PREDICTING END")
    pred_var=np.empty([len(target_test),],dtype=int)
    for i in range(0,len(pred)):
        pred_var[i]=np.argmax(pred[i])
    return (pred_var,pred)
    
    
def evaluate(y_pred,y):
    print("\n\n============Evaluation metrices with five trained Categories...================")
    logging.info('Evaluating Model for Accuracy')
    acc_model = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y,y_pred)
    print("Accuracy : ", (acc_model)*100)
    print("\nKappa: ",kappa)
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))
    
def evaluate_other_category(y_pred,y,y_pred_proba):
    print("\n\n============Evaluation metrices with OTHER Category...================")
    y_new = np.full(len(y_pred),5)
    # check threshold and decide to keep model's class or "other" class(5)
    for i in range(0,len(y_pred)):
        if y_pred_proba[i,y_pred[i]] > 0.3 :
            y_new[i] = y_pred[i]
        
    print("News of Other category = ",(y_new==5).sum())
    acc_rfc = accuracy_score(y, y_new)
    kappa = cohen_kappa_score(y,y_new)
    print("Accuracy including other category: ", acc_rfc*100)
    print("\nKappa of other category: ",kappa)
    print(confusion_matrix(y, y_new))
    print(classification_report(y, y_new))
    
def testing(X,y,choiceModel, modelpath):
    logging.info('Started Testing ')
    tStart=time.time()
    if choiceModel != 7:
        xtest_tfidf = feature_extraction(X,choiceModel, modelpath)
        y_pred,y_pred_proba = predict(xtest_tfidf,y,choiceModel, modelpath)
        evaluate(y_pred,y)
        evaluate_other_category(y_pred,y,y_pred_proba)
    else:
        X=X.values.tolist()
        y_pred,y_pred_proba = predict_dl(X,y,choiceModel, modelpath)
        evaluate(y_pred,y)
        #evaluate_other_category(y_pred,y,y_pred_proba)
    tEnd=time.time()
    logging.info('Finished Testing , time taken: ', tEnd - tStart)
    
    