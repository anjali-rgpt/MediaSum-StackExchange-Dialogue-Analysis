import preprocess
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.evaluation.measures import CoherenceCV
import spacy
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk import sent_tokenize

upload_folder = 'static/outputs/'

lemmatizer_model = spacy.load('en_core_web_sm', disable = ['ner', 'parser'])
pos_tags = ['NN', 'VB', 'JJ']
stopword_eng = stopwords.words('english')
contractions_dict = {"aren t" : "are not", "can t" : "cannot", "couldnt":"could not", "couldn t" : "could not", "didn t" : "did not", "doesn t" : "does not", "don t" : "do not", "haven t" : "have not", "hasn t" : "has not" , "hadn t" : "had not", "i m" : "i am", "i ve" : "i have", "isn t" : "is not", "it's" : "it is" , "mustn t" : "must not", "shouldn t" : "should not", "wasn t" : "was not", "weren t" : "were not", "wouldn t" : "would not", "won t":"will not", "you re" : "you are", "you ll" : "you will" , "we ll" : "we will", "you ve" : "you have", "we ve" : "we have"}

def ngrams(data, column = 'summary'):
    assert type(data) == type(pd.DataFrame()), "Data is not a dataframe"
    assert column in data.columns, "No column called summary"

    count_words = np.sum(data[column].apply(lambda x: preprocess.word_count(x)))
    # print("Total word count:", count_words)

    data['Normalized'] = data[column].apply(lambda x: preprocess.create_pipeline(x, ['ctrc', 'norm', 'tags', 'stop', 'lemm', 'norm'], word_length = 3, contractions = contractions_dict, stopwords = stopword_eng, lemmatizer = lemmatizer_model, relevant_tags = pos_tags, general_category = True))
    trigrams, _ = preprocess.generate_ngrams(data, 'Normalized', n = 3, frequency = int(count_words*0.01)+1, best_number = 25, stopwords_list = stopword_eng)
    if type(trigrams) != type([]):
        trigrams = [' '.join(gram) for gram in trigrams.values]
    else:
        trigrams = []

    bigrams, _ = preprocess.generate_ngrams(data, 'Normalized', n = 2, frequency = int(count_words*0.01)+1, best_number = 25, stopwords_list = stopword_eng)
    if type(bigrams) != type([]):
        bigrams = [' '.join(gram) for gram in bigrams.values]
    else:
        bigrams = []

    return trigrams + bigrams

def transcript_id(data):
    return data['id']

def ctm_model(data, column = 'utt'):   
    print("\nNEW ITEM")
    preprocessed_list = []
    
    if type(data) == type(pd.Series([1,2,3])):
        data = pd.DataFrame(data).transpose()
    # print(data[column])
    ngrams_chosen = ngrams(data)
    print(type(data[column]))
    if type(data[column].iloc[0]) == type([]):
        print(data[column])
        to_process = data[column].iloc[0]
    else:
        to_process = sent_tokenize(data[column].tolist()[0])
    for item in to_process:
        # print(item, type(item))
        preprocessed_list.append(' '.join(preprocess.create_pipeline(item, ['tags', 'ctrc', 'norm', 'stop', 'lemm', 'norm'], word_length = 3, contractions = contractions_dict, stopwords = stopword_eng, lemmatizer = lemmatizer_model, relevant_tags = ['NN', 'VB'], general_category = False, ngrams_list = ngrams_chosen)))

    print(preprocessed_list)
    to_dictionary = [string.split() for string in preprocessed_list]
    tp = TopicModelDataPreparation("paraphrase-distilroberta-base-v1")

    training_dataset = tp.fit(text_for_contextual= to_process, text_for_bow = preprocessed_list)
    topics_per_document = [2, 3, 4, 5]
    best_num = 0
    best_coherence = -9999
    scores_lda = []
    for num_topic in topics_per_document:
        ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=num_topic, num_epochs=20)
        ctm.fit(training_dataset)
        coh_model = CoherenceCV(topics = ctm.get_topic_lists(), texts = to_dictionary)
        coherence_score = coh_model.score()
        scores_lda.append(coherence_score)
        print("\nNumber of topics:", num_topic)
        print("Coherence score:", coherence_score)
        if coherence_score > best_coherence:
            best_coherence = coherence_score
            best_num = num_topic
            print("Best coherence so far")
    print("Best number of topics:", best_num)
    ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=best_num, num_epochs=10)
    ctm.fit(training_dataset)
    return ctm, best_num

def get_ctm_topics(model, parameter = 10):
    return model.get_topic_lists(parameter)

def generate_word_cloud(data, model, parameter = 10):
    imagelists = []
    topic_lists = get_ctm_topics(model, parameter)
    for num_topic in range(len(topic_lists)):
        #plt.title("Topic Cluster " + str(num_topic))
        wc = WordCloud().generate(' '.join(topic_lists[num_topic]))
        path_val = os.path.join(upload_folder, str(data['id']) + '_Topic_' + str(num_topic) + '.png').replace('%5C','/')
        print(path_val)
        wc.to_file(path_val)
        imagelists.append(str(data['id']) + '_Topic_' + str(num_topic)+'.png')
        # plt.imshow(wc)
        #plt.axis("off")
        #plt.show()
    return imagelists




