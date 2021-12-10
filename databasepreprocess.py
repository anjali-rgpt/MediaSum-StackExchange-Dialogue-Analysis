import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models import Word2Vec, FastText
from scipy import spatial

regular_expression = re.compile('[' + re.escape('!@#$^&*\'()+=-_,./:;<>?"[\\]^_`{|}~')+'\\r\\t\\n]')
stopwords_eng = stopwords.words('english')

def load_into_pandas(main_directory, lines_arg = True):
    assert type(main_directory) == type('Document'), "Path is not a string."
    print(main_directory)

    r = []                                                                                                            
                                                                             
    for root, subdirs, files in os.walk(main_directory):
        for file in files:
            # print(file)
            if file[-5:]=='.json':
                r.append(os.path.join(root, file))

    df_total = pd.concat([pd.read_json(filename, lines = lines_arg) 
                              for filename in r])
    df_total.reset_index(inplace = True)
    
    return df_total

def preprocess_data(data, remove_characters = regular_expression, stopwords_list = stopwords_eng):    
    assert type(data) == type('Document'), "Data is not a string"
    assert type(remove_characters) == re.Pattern, "Characters to remove are not a regex Pattern object."
    
    temp_data = data.lower()
    temp_data = remove_characters.sub(' ', temp_data)
    split_list= temp_data.split()
    # print(split_list)
    temp_data = [word for word in split_list if word not in stopwords_list]
    return temp_data

def get_cosine_simlarity(record):
    v1 = record.iloc[0:32]
    v2 = record.iloc[32:64]
    return spatial.distance.cosine(v1, v2)

def find_common_words(record):
    
    q1 = set(record['preprocessed_q1'])
    q2 = set(record['preprocessed_q2'])
    return q1 & q2

def find_total_words(record):
        
    q1 = set(record['preprocessed_q1'])
    q2 = set(record['preprocessed_q2'])
    return len(q1) + len(q2)
    
def find_shared_ratio(record):
    
    intersect_words = 0
    if len(record['shared_words']) > 0:
        intersect_words = len(record['shared_words'])
    return intersect_words / record['total_words']

def retrieve_db():    
    try:
        database = pd.read_csv('database.csv')
        extras = load_into_pandas('pythia\\stack_exchange_data\\', lines_arg = True).loc[:, ['cluster_id', 'order']]
        other_columns = pd.read_csv('database_encoded.csv')
        print("Loaded.")
        return database, other_columns, extras

    except:
        out = load_into_pandas('pythia\\stack_exchange_data\\', lines_arg = True)
        db_new = pd.DataFrame(out, columns = out.columns)

        db_new['qid1'] = out['post_id']
        db_new['preprocessed_q1'] = out['body_text'].apply(lambda x: preprocess_data(x))
        extras = db_new.loc[:,['cluster_id', 'order']]
        db_new.drop(['cluster_id', 'order'], axis = 1, inplace = True)
        db_new.to_csv('database.csv')
        print("Created database")

        model = FastText.load('models\\fasttextmodel.model')

        document_vectors_q1 = pd.DataFrame()
        for document in db_new['preprocessed_q1']:
            temp_vector = pd.DataFrame()
    
            for word in document:
                embedding = model.wv[word]
                temp_vector = temp_vector.append(pd.Series(embedding), ignore_index = True)
            current_vector = temp_vector.mean()
            document_vectors_q1 = document_vectors_q1.append(current_vector, ignore_index = True)
        document_vectors_q1.to_csv('database_encoded.csv')
        print(document_vectors_q1.shape)
        print("Encoded.")
        return db_new, document_vectors_q1, extras

