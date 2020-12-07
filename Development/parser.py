import sys
import pandas as pd
import csv
import sklearn
import numpy as np
from utils import *
from unidecode import unidecode
import dateparser
from datetime import date
from numpy import mean
from numpy import std
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

def parse_data(file_name):

    read_file = pd.read_excel(file_name)
    read_file.to_csv(file_name+'.csv', index = None, header = True)
    df = pd.DataFrame(pd.read_csv(file_name+'.csv'))

    # to_lower everything
    df = df.apply(lambda x : x.astype(str).str.lower())
    
    # remove all diacritics
    for col in df.columns:
        df[col] = df[col].apply(unidecode)

    # **************************************************************************

    # drop 'institutia sursa' column     
    df.drop('instituția sursă', axis = 1, inplace = True)

    # drop 'mijloace de transport folosite' column
    df.drop('mijloace de transport folosite', axis = 1, inplace = True)

    # encode sex column
    df['sex'] = np.where(df['sex'].str.startswith('masc' or 'm'), 'masculin', 'feminin')

    # encode age
    age_bins= [0, 18, 29, 39, 49, 64, 74, 84, 100]
    age_intervals_labels = ['0-18', '19-29', '30-39', '40-49', '50-64', '65-74', '75-84', '85-100']
    df['vârstă'] = np.where(df['vârstă'].str.isnumeric(), df['vârstă'], -1)
    df['vârstă'] = pd.to_numeric(df['vârstă'])
    df['vârstă'] = pd.cut(df['vârstă'], bins = age_bins, labels = age_intervals_labels, right = True)
    df['vârstă'] = df['vârstă'].cat.add_categories('unknown').fillna('unknown')
    
    # encode 'simptome declarate' column
    value = []
    for s in df['simptome declarate']:
        number = len([word for word in simptoms if word in str(s)])
        if(number < 5):
            value.append(number)
        else:
            value.append('5+')
            

    df['simptome declarate'] = value


    # encode 'simptome raportate la internare' column
    value = []
    for s in df['simptome raportate la internare']:
        number = len([word for word in simptoms if word in str(s)])
        if(number < 5):
            value.append(number)
        else:
            value.append('5+')

    df['simptome raportate la internare'] = value


    # encode 'rezultat testare' column
    value = []
    for r in df['rezultat testare']:
        if str(r).startswith('poz'):
            value.append('pozitiv')
        else:
            value.append('negativ')
    df['rezultat testare'] = value


    # encode 'confirmare contact cu o persoană infectată' column
    df['confirmare contact cu o persoană infectată'] = np.where(
        df['confirmare contact cu o persoană infectată'].str.contains('da'), 
        'da', 'nu'
    )


    # encode 'diagnostic și semne de internare' column
    value = []
    i = -1
    for s in df['diagnostic și semne de internare']:
        i += 1
        number = 0
        number += len([word for word in diagnostic_simptoms_lower if word in str(s)])
        number += 2 * len([word for word in diagnostic_simptoms_med if word in str(s)])
        number += 7 * len([word for word in diagnostic_simptoms_higher if word in str(s)])
        if(number < 10):
            value.append(number)
        else:
            value.append('10+')
    df['diagnostic și semne de internare'] = value


    # encode 'istoric de călătorie' column
    value = []
    for s in df['istoric de călătorie']:
        if len([word for word in countries if word in str(s)]) > 0:
            value.append('da')
        else:
            value.append('nu')
    df['istoric de călătorie'] = value



    # encode 'dată debut simptome declarate' column
    value = []
    for s in df['dată debut simptome declarate']:
        if s == 'nan' or len(s) < 6 or ('nu' in s):
            value.append(0)
        else:
            parsed_data = dateparser.parse(s)
            if parsed_data is None:
                value.append(0)
            else:
                value.append(parsed_data)

    df['dată debut simptome declarate'] = value


    # encode 'data rezultat testare' column
    value = []
    for s in df['data rezultat testare']:
        if s == 'nan' or len(s) < 6 or ('nu' in s):
            value.append(0)
        else:
            parsed_data = dateparser.parse(s)
            if parsed_data is None:
                value.append(0)
            else:
                value.append(parsed_data)
    
    df['data rezultat testare'] = value


    # encode 'perioada intre data rezultat testare si data debut'
    value = []
    i = -1
    for s in df['data rezultat testare']:
        i += 1
        date_start = df['dată debut simptome declarate'][i] 
        date_result = s
        if date_start == 0 or date_result == 0:
            value.append('unknown')
        else:
            delta = date_result - date_start
            if delta.days < 0:
                value.append('unknown')
                #print('Zile = {} --- Final = {}  ---- Inceput = {}'.format(delta.days, date_result, date_start))
            else:
                if delta.days >= 0 and delta.days <= 14:
                    value.append('0-2 saptamani')
                elif delta.days > 14 and delta.days <= 21:
                    value.append('2-3 saptamani')
                else:
                    value.append('> 3 saptamani')

    # de fapt e perioada intre data rezultat testare si data debut
    df['dată internare'] = value

    # drop 
    df.drop('data rezultat testare', axis = 1, inplace = True)
    
    # drop 
    df.drop('dată debut simptome declarate', axis = 1, inplace = True)

    
    # *********************************************************************************************
    # SOME MACHINE LEARNING 
    
    df.to_csv(file_name+'_clean.csv', index = True, quoting = csv.QUOTE_ALL)
    dataset = pd.read_csv(file_name+'_clean.csv', index_col = 0)
    data = dataset.values
    return data

def IA(data,data2):
    X = data[:, :-1].astype(str)
    y = data[:, -1].astype(str)

    X2 = data2[:,:-1].astype(str)
    y2 = data2[:, -1].astype(str)

    # split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 1)
    # ordinal encode input variables
    onehot_encoder = OneHotEncoder(handle_unknown = 'ignore')
    onehot_encoder.fit(X_train)
    X_train = onehot_encoder.transform(X_train)
    X_test = onehot_encoder.transform(X_test)
    # ordinal encode target variable
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
    
    #define the model
    model = LogisticRegression()
    # fit on the training set
    model.fit(X_train, y_train)
    # predict on test set
    yhat = model.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, yhat)
    print('Accuracy for test-data: %.2f' % (accuracy * 100))

    ####################################
    X2_test = onehot_encoder.transform(X2)
    y2_test = label_encoder.transform(y2)

    yhat2 = model.predict(X2_test)
    accuracy = accuracy_score(y2_test, yhat2)
    print('Accuracy on custom input : %.2f' % (accuracy * 100))

    results = confusion_matrix(y2_test, yhat2)
    print ('Confusion Matrix :')
    print(results)
    print ('Accuracy Score is',accuracy_score(y2_test, yhat2))
    print ('Classification Report : ')
    print (classification_report(y2_test, yhat2))
    print('AUC-ROC:',roc_auc_score(y2_test, yhat2))
    print('LOGLOSS Value is',log_loss(y2_test, yhat2))


if __name__ == "__main__":
    data = parse_data('mps.dataset.xlsx')
    data2 = parse_data('')
    IA(data,data2)
    