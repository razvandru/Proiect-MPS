import sys
import pandas as pd
import sklearn
import numpy as np

if __name__ == "__main__":

    read_file = pd.read_excel('mps.dataset.xlsx')
    read_file.to_csv('mps.dataset.csv', index = None, header = True)
    df = pd.DataFrame(pd.read_csv("mps.dataset.csv"))

    
    # a = pd.get_dummies(df, columns = ['sex']).head()
    # print(a)

    # encode sex column
    df['sex'] = np.where(df['sex'].str.startswith('MASC' or 'masc' or 'M' or 'm'), 'masculin', 'feminin')
    a = pd.get_dummies(df, columns = ['sex']).head()

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
    
    #print(df['simptome declarate'].value_counts().keys().tolist())
    simptoms = ['tuse', 'dispnee', 'febra', 'frison', 'frisoane', 'temperatura', 'gust', 'miros', 'cefalee']
    #df['vârstă'] = np.where(df['simptome declarate'].str., df['vârstă'], -1)

    
    # frequency = len([word for words in simptoms if word in df['simptome declarate']])
    # df['simptome declarate'] = np.where(frequency > 0, frequency, 0)




    


    

    