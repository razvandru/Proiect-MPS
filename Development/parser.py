import sys
import pandas as pd
import sklearn
import numpy as np
from utils import *
from unidecode import unidecode

if __name__ == "__main__":

    read_file = pd.read_excel('mps.dataset.xlsx')
    read_file.to_csv('mps.dataset.csv', index = None, header = True)
    df = pd.DataFrame(pd.read_csv("mps.dataset.csv"))

    # to_lower everything
    df = df.apply(lambda x : x.astype(str).str.lower())
    
    # remove all diacritics
    for col in df.columns:
        df[col] = df[col].apply(unidecode)

    # **************************************************************************

    # encode sex column
    df['sex'] = np.where(df['sex'].str.startswith('masc' or 'm'), 'masculin', 'feminin')
    pd.get_dummies(df, columns=['sex'], prefix = '', prefix_sep = '')

    # drop 'institutia sursa' column     
    df.drop('instituția sursă', axis = 1, inplace = True)

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
        value.append(len([word for word in simptoms if word in str(s).lower()]))

    df['simptome declarate'] = value


    # encode 'simptome raportate la internare' column
    value = []
    for s in df['simptome raportate la internare']:
        value.append(len([word for word in simptoms if word in str(s).lower()]))

    df['simptome raportate la internare'] = value


    # encode 'rezultat testare' column
    value = []
    for r in df['rezultat testare']:
        if str(r).startswith('neg'):
            value.append('negativ')
        elif str(r).startswith('poz'):
            value.append('pozitiv')
        else:
            value.append('neconcludent')
    df['rezultat testare'] = value
    pd.get_dummies(df, columns=['rezultat testare'], prefix = '', prefix_sep = '')


    # encode 'confirmare contact cu o persoană infectată' column
    df['confirmare contact cu o persoană infectată'] = np.where(
        df['confirmare contact cu o persoană infectată'].str.contains('da'), 
        'da', 'nu'
    )
    pd.get_dummies(df, columns=['confirmare contact cu o persoană infectată'], prefix = '', prefix_sep = '')
    
    #print(df['confirmare contact cu o persoană infectată'].value_counts())
    


    

    