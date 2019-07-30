# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 17:34:25 2019

@author: dbda

Functions for Training Model

"""
import logging
import time
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from joblib import dump
from textpackage import data_preprocess as pp
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
import numpy as np
from keras.layers import Activation, Conv1D, Dense, Embedding, Flatten, Input, MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

modelpath = 'F:/PROJECT/Deploy2/'
model_names={1:'SGD',2:'RF',3:'KNN',4:'MNB',5:'VOTING',6:'STACK'}
def save_vocab(vocab,name,modelpath):
    pickle_out = open(modelpath + 'vocabularies/'+name+'.pickle',"wb")
    print(modelpath + 'vocabularies/'+name+'.pickle')
    pickle.dump(vocab, pickle_out)
    pickle_out.close()



def feature_extraction(X,choiceModel,modelpath):
    print("\n\n============Extracting features================")
    logging.info('Started feature Extraction ')
    tfidf_vect = TfidfVectorizer(analyzer='word',min_df=3, token_pattern=r'[a-zA-Z]{2,}', max_features=50000, stop_words = 'english')
    tfidf_vect.fit(X)
    save_vocab(tfidf_vect.vocabulary_,model_names.get(choiceModel),modelpath)
    xtrain_tfidf =  pd.DataFrame(tfidf_vect.transform(X).todense(),columns=tfidf_vect.get_feature_names())
    logging.info('Finished feature Extraction ')
    return xtrain_tfidf

def stack_ensembling(xtrain_tfidf,y,choiceModel, modelpath):
    knn = KNeighborsClassifier(algorithm='auto',leaf_size=1, n_neighbors=7,weights = 'uniform',  n_jobs=-1  )
    knn.fit(xtrain_tfidf, y)
    knn_pred = knn.predict(xtrain_tfidf)
    knn_pred = pd.get_dummies(knn_pred, drop_first = True)
    dump(knn, modelpath + 'STACK_KNN.joblib') 
    
    rfc = RandomForestClassifier(min_samples_leaf=10, min_samples_split=50, n_estimators=550,random_state=2019)
    rfc.fit(xtrain_tfidf, y)# Model Building
    rfc_pred = rfc.predict(xtrain_tfidf) # Applying built on test data
    rfc_pred = pd.get_dummies(rfc_pred, drop_first = True)
    dump(rfc, modelpath + 'STACK_RFC.joblib') 
    
    sgdc = SGDClassifier(alpha=0.001, loss='modified_huber', max_iter=50, penalty='l2')
    sgdc.fit(xtrain_tfidf, y)
    sgdc_pred = sgdc.predict(xtrain_tfidf)
    sgdc_pred = pd.get_dummies(sgdc_pred, drop_first = True)
    dump(sgdc, modelpath + 'STACK_SGDC.joblib') 

    X_pred = pd.concat([knn_pred, rfc_pred, sgdc_pred], axis=1)
    
    lr = LogisticRegression()
    lr.fit(X_pred, y)
    dump(lr, modelpath + 'STACK_LEVEL2_LR.joblib') 






def build_model(xtrain_tfidf,y,choiceModel, modelpath):
    print("\n\n============Building Model...================")
    logging.info('Started Model Building')
    if choiceModel==1:
        model = SGDClassifier(alpha=0.001, loss='modified_huber', max_iter=50, penalty='l2')
        model.fit(xtrain_tfidf, y)
#        dump(model, r'F:\PROJECT\Deploy2\Models\SGD_CLASSIFIER.joblib') 
        dump(model, modelpath + 'Models/SGD_CLASSIFIER.joblib') 
    elif choiceModel==2:
        model = RandomForestClassifier(min_samples_leaf=10, min_samples_split=50, n_estimators=550,random_state=2019)
        model.fit(xtrain_tfidf, y)
        #filename=pp.file_save()
        #dump(rfc, filename)  
        dump(model, modelpath + 'Models/RANDOM_FOREST.joblib')  
    elif choiceModel==3:
        model = KNeighborsClassifier( algorithm='auto', leaf_size=1, n_neighbors=7, weights = 'uniform', n_jobs=-1)
        model.fit(xtrain_tfidf, y)
        dump(model, modelpath + 'Models/K_NN_CLASSIFIER.joblib')
    elif choiceModel==4:
        
        model = MultinomialNB()
        model.fit(xtrain_tfidf, y)
        dump(model, modelpath + 'Models/MULTINOMAIL_NAIVE_BAYES.joblib') 
    elif choiceModel==5:
        sgd_clf = SGDClassifier(alpha=0.001, loss='modified_huber', max_iter=50, penalty='l2')
        rfc = RandomForestClassifier(min_samples_leaf=10, min_samples_split=50, n_estimators=550,random_state=2019)
        knn = KNeighborsClassifier(algorithm='auto',leaf_size=1, n_neighbors=7,weights = 'uniform',  n_jobs=-1 )
        model = VotingClassifier(estimators=[('RFC',rfc),
                                              ('SGDC',sgd_clf),
                                              ('KNN',knn)], voting='soft')
        model.fit(xtrain_tfidf, y)
        dump(model, modelpath + 'Models/VOTING_CLASSIFIER.joblib')
    elif choiceModel==6:
        stack_ensembling(xtrain_tfidf,y,choiceModel, modelpath)
    logging.info('Finished Model Building')





def build_model_dl(texts,target,modelpath):
    logging.info('Started Model Building')
    vocab_size = 20000
    tokenizer = Tokenizer(num_words=vocab_size) # Setup tokenizer
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts) # Generate sequences
    word_index = tokenizer.word_index
    #print('Found {:,} unique words.'.format(len(word_index)))
    max_length = 300
    embedding_dim = 300 # We use 100 dimensional glove vectors
    data = pad_sequences(sequences, maxlen=max_length)
    labels = to_categorical(np.asarray(target))
    #print('Shape of data:', data.shape)
    #print('Shape of labels:', labels.shape)
    glove_dir = r'F:\PROJECT\20-newsgroup-sklearn\glove.6B'
    embeddings_index = {} 
    with open(os.path.join(glove_dir, 'glove.6B.300d.txt' ),encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0] # The first value is the word, the rest are the values of the embedding
            embedding = np.asarray(values[1:], dtype='float32') # Load embedding
            embeddings_index[word] = embedding # Add embedding to our embedding dictionary
    print('Found {:,} word vectors in GloVe.'.format(len(embeddings_index)))
    word_index = tokenizer.word_index
    nb_words = min(vocab_size, len(word_index)) # How many words are there actually
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    
    for word, i in word_index.items():
        if i >= vocab_size: 
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
    
    model = Sequential()
    model.add(Embedding(vocab_size, 
                        embedding_dim, 
                        input_length=max_length, 
                        weights = [embedding_matrix], 
                        trainable = False))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
    
    model.fit(data, labels, validation_split=0.2, epochs=10)
    dump(model, modelpath + 'Models/DEEP_LEARNING.joblib')
    dump(tokenizer,modelpath + 'Models/DEEP_LEARNING_TOKENIZER.joblib')
    logging.info('Finished Model Building')
    
    
    
def training(X,y,choiceModel, modelpath):
    logging.info('Started Training the Model')
    tStart=time.time()
    if choiceModel != 7:
        xtrain_tfidf=feature_extraction(X,choiceModel,modelpath)
        build_model(xtrain_tfidf,y,choiceModel, modelpath)
    else:
        X=X.values.tolist()
        build_model_dl(X,y,modelpath)
    tEnd=time.time()
    logging.info('Finished Training , time taken: ', tEnd - tStart)

