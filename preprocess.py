import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
import re


regular_expression = re.compile('[' + re.escape('!@#$^&*()+=-_,./:;<>?"[\\]^_`{|}~')+'\\r\\t\\n]')

def turn_count(data):   
    assert type(data) == type([]), "Data is not a list."  
    return len(data)

def word_count(data, special_characters_rm = regular_expression):
    
    assert type(data) == type([]) or type(data) == type('Document'), "Data is not a list or string"
    assert type(special_characters_rm) == re.Pattern, "Your special characters are not a regex Pattern object"    
    wordcount = 0
    if type(data) == type([]):
        wordcount = 0
        for speaker_turn in data:
            sentences = sent_tokenize(speaker_turn)
            for sentence in sentences:
                de_punct_sentence = re.sub(special_characters_rm, ' ', sentence)
                wordcount += len(de_punct_sentence.split())

        return wordcount   
    else:
        wordcount = 0
        sentences = sent_tokenize(data)
        for sentence in sentences:
            de_punct_sentence = re.sub(special_characters_rm, ' ', sentence)
            wordcount += len(de_punct_sentence.split())
            
    return wordcount

def speaker_count(data):
    
    assert type(data) == type([]), "Data is not a list."
    
    unique_values = pd.Series(data)
    unique_set = set(unique_values)
    for speaker in unique_values:
        name = speaker.split()
        if len(name) == 1:
            for other_speaker in list(unique_set - {name[0]}):
                if name[0] in other_speaker:
                    unique_set = unique_set - {other_speaker}
            
    return len(unique_set)
    
def normalize_data(data, keep_nonascii = False, keep_punctuation = False, remove_extra_spaces = True, lowercase = True, word_length = None):
    """
    Performs specific cleaning operations on the data, as long as the data is a string.
    :param data: string representing review
    :param keep_nonascii: boolean value representing whether to keep non-ascii characters or not
    :param keep_punctuation: boolean value representing whether to keep punctuation or not
    :param remove_extra_spaces: boolean value representing whether to remove extra spaces between, before, and after words
    :param lowercase: boolean value representing whether to lowercase data or not
    :param word_length: if None, word length is not taken into account. If it is a number, only words greater than or equal to specific length will be kept.
    :return: normalized string
    """
    assert type(data) == type('Document'),"Your data is not a string."
    temp_data = data
    if not keep_nonascii:
        temp_data = temp_data.encode('ascii', errors = 'ignore')
    if not keep_punctuation:
        regular_expression = re.compile('[' + re.escape('!@#$^&*\'()+=-_,./:;<>?"[\\]^_`{|}~')+'0-9\\r\\t\\n]')
        temp_data = regular_expression.sub(' ', temp_data.decode('utf-8'))
    if remove_extra_spaces:
        split_data = temp_data.split(' ')
        just_words = [word.strip() for word in split_data if word not in ['','\r','\t',' ']]
        temp_data = ' '.join(split_data)
    if lowercase:
        temp_data = temp_data.lower()
    if word_length:
        assert type(word_length) == type(5)
        split_data = temp_data.split(' ')
        select_words = [word for word in split_data if len(word)>=word_length]
        temp_data = ' '.join(split_data)
        
    return temp_data

def tokenize(data):
    """
    Function to tokenize data. Best used AFTER stopword removal, contraction replacement, and normalization. 
    Best used before lemmatization and POS tagging.
    :param data: string representing review
    :return: tokenized string as list of tokens
    """
    assert type(data) == type('Document'), "Your data is not a string."
    return word_tokenize(data)

def replace_contractions(data, contractions):
    """
    Function to replace contractions in data. Best used BEFORE normalization.
    :param data: string representing review
    :param contractions: dictionary of predefined contractions and their expansions
    :return: string with contractions replaced
    """
    assert type(data) == type('Document'), "Your data is not a string."
    assert type(contractions) == type({}), "Your contractions are not in a dictionary."
    
    contraction_list = contractions.keys()
    temp_data = str(data.lower()).split()
    for i in range(len(temp_data)):
        if temp_data[i] in contraction_list:
            temp_data[i] = contractions[temp_data[i]]
    data = ' '.join(temp_data)
    return data

def lemmatize(data, lemmatizer):
    """
    Function to lemmatize data. Best used BEFORE tokenization and AFTER normalization.
    :param data: string representing review
    :param lemmatizer: NLTK/Spacy lemmatizer object
    :return: string with lemmas only
    """
        
    assert type(data) == type('Document'), "Your data is not a string."
    lemmas = []
    sentences = sent_tokenize(data)
    for s in sentences:
        processed_sentence = lemmatizer(s)
        lemmas.extend([word.lemma_ for word in processed_sentence])
    return ' '.join(lemmas)


def clean_pos_tags(data, relevant_tags, general_category = True):
    """
    Function to remove unnecessary words by filtering out based on part of speech. 
    Best used AFTER normalization and contraction replacement, but BEFORE other steps.
    :param data: string representing review
    :param relevant_tags: list of NLTK POS tags to keep
    :param general_category: if True, you only need to specify general categories like JJ, NN, and it will take subcategories too.
    :return: string with required words only
    """
    assert type(data) == type('Document'), "Your data is not a string."
    assert type(relevant_tags) == type([]), "Your POS tags are not in a list."
    
    tagged_data = nltk.pos_tag(data.split())
    # print(tagged_data)
    if general_category:
        kept_words = [tag_pair[0] for tag_pair in tagged_data if tag_pair[1].startswith(tuple(relevant_tags))]
    else:
        kept_words = [tag_pair[0] for tag_pair in tagged_data if tag_pair[1] in relevant_tags]
    return ' '.join(kept_words)

def remove_stopwords(data, stopwords): 
    """
    Function to remove stopwords from data. Best used AFTER normalization.
    :param data: string representing review
    :param stopwords: list of stopwords to be removed
    :return: string with stopwords removed
    """

    assert type(data) == type('Document'), "Your data is not a string."
    assert type(stopwords) == type([])

    split_data = data.split(' ')
    cleaned_data = [word for word in split_data if word not in stopwords and len(word)>3]
    new_data = ' '.join(cleaned_data)
    return new_data

def replace_multi_words(data, multi_dict):
    
    """
    Function to replace common multi-words in data.
    :param data: string representing review
    :param multi_dict: dictionary representing mutli-words to combine.
    :return: string with replaced multi-words
    """
    
    assert type(data) == type('Document'), "Your data is not a string."
    assert type(multi_dict) == type({}), "Your multi-word dictionary is not a dictionary."
    
    new_data = str(data)
    for replacement in multi_dict.keys():
        if re.search(replacement, data):
            new_data = re.sub(replacement, multi_dict[replacement], new_data)
    return new_data

def replace_ngrams(data, ngrams_list):
    """
    Function to replace ngrams in data.
    :param data: string representing review
    :param ngrams_list: list representing ngrams
    :return: string with replaced ngrams
    """
    
    assert type(data) == type('Document'), "Your data is not a string."
    assert type(ngrams_list) == type([]), "Your list of n-grams is not a list."
    
    new_data = str(data)
    for replacement in ngrams_list:
        new_data = re.sub(replacement, '_'.join(replacement.split()), new_data)
    return new_data 

def create_pipeline(data, 
                    order, 
                    keep_nonascii = False, 
                    keep_punctuation = False, 
                    remove_extra_spaces = True, 
                    lowercase = True, 
                    word_length = None,
                    contractions = None,
                    lemmatizer = None,
                    stopwords = None,
                    relevant_tags = None,
                    general_category = True,
                    multi_dict = None,
                    ngrams_list = None
                    
                   ):
    """
    Function to set up a preprocessing pipeline on data. Tokenization is done at the end by default.
    :param data: string representing review
    :param order: list representing order of operations.
    :param keep_nonascii: boolean value representing whether to keep non-ascii characters or not
    :param keep_punctuation: boolean value representing whether to keep punctuation or not
    :param remove_extra_spaces: boolean value representing whether to remove extra spaces between, before, and after words
    :param lowercase: boolean value representing whether to lowercase data or not
    :param word_length: if None, word length is not taken into account. If it is a number, only words greater than or equal to specific length will be kept.
    :param contractions: dictionary of predefined contractions and their expansions
    :param lemmatizer: NLTK/Spacy lemmatizer object
    :param relevant_tags: list of NLTK POS tags to keep
    :param general_category: if True, you only need to specify general categories like JJ, NN, and it will take subcategories too.
    :param multi_dict: dictionary representing mutli-words to combine.
    :param ngrams_list: list representing ngrams
    
    *Allowed operations: 'norm'-> normalization, 'stop'->stopword removal, 'ctrc'->contraction replacement, 'lemm'->lemmatization, 'tags'->cleaning POS tags, 'mult' -> replacing multi words, 'ngrm' -> replacing ngrams.
    :return: list of words after preprocessing
    """
    assert type(data) == type('Document') or type(data) == type([]), "Your data is not a string or list."
    assert type(order) == type([]), "Your operations order is not a list."
    
    if type(data) == type([]):
        data = ' '.join(data)
    allowed_ops = ['norm', 'stop', 'ctrc', 'lemm', 'tags', 'mult', 'ngrm']
    
    pro_data = data
    
    for operation in order:
        assert operation in allowed_ops, "Operation not in allowed list."
        if operation == 'norm':
            pro_data = normalize_data(pro_data, keep_nonascii, keep_punctuation, 
                    remove_extra_spaces, 
                    lowercase, 
                    word_length)
            # print(pro_data)
            
        elif operation == 'stop':
            assert stopwords != None, "No stopwords specified"
            pro_data = remove_stopwords(pro_data, stopwords)
            # print(pro_data)
        
        elif operation == 'ctrc':
            assert contractions != None, "No contractions specified"
            pro_data = replace_contractions(pro_data, contractions)
            #print(pro_data)
            
        elif operation == 'lemm':
            assert lemmatizer != None, "There is no lemmatizer specified"
            pro_data = lemmatize(pro_data, lemmatizer)
            # print(pro_data)
            
        elif operation == 'tags':
            assert relevant_tags != None, "There are no tags specified"
            pro_data = clean_pos_tags(pro_data, relevant_tags, general_category)
            # print(pro_data)
            
        elif operation == 'mult':
            assert multi_dict != None, "There is no multi-word dictionary specified"
            pro_data = replace_multi_words(pro_data, multi_dict)
            # print(pro_data)
            
        elif operation == 'ngrm':
            assert ngrams_list != None, "There are no n-grams specified"
            pro_data = replace_ngrams(pro_data, ngrams_list)
    
    
    return tokenize(pro_data)

def generate_ngrams(data, column, n = 2, frequency = 20, best_number = 10, min_pmi = 4, stopwords_list = stopwords.words('english')):
    """
    Function to generate the most common co-occuring phrases in the data.
    :param data: dataframe representing reviews list
    :param column: column name to consider
    :param n: integer representing the number of words to consider for an n-gram. Can take values 2 or 3.
    :param frequency: integer value showing the minimum frequency of n-gram to be considered.
    :param best_number: value determining how many n_grams to return
    :param min_pmi: minimum pmi value to keep
    :param stopwords_list: sequence of stopwords
    :return: list of n-grams for the data
    """
    assert type(data) == type(pd.DataFrame(columns = ['x'])), "Your data is not a dataframe."
    assert type(column) == type('Document'), "Your data is not a string."
    assert column in data.columns, "Invalid column name"
    assert type(n) == type(int(5)), "Your n value is not an integer."
    assert n in [2,3], "Incorrect value of n."
    assert type(frequency) == type(int(1)), "Your frequency value is not an integer."
    assert frequency > 0, "Your frequency does not fall into the appropriate range."
    assert type(min_pmi) == type(int(1.0)), "Your minimum PMI is not a floating point number."
    assert min_pmi > 0, "Your minimum PMI does not fall into the appropriate range."
    assert type(best_number) == type(int(5)) and best_number>0, "Best number is not an integer greater than 0"
    assert type(stopwords_list) in [type([]), type({'x':1}.keys())], "Stopwords are not in the right format"
    
    final_list = []
    
    
    if n == 2:
        bigram_collocations = nltk.collocations.BigramAssocMeasures()
        total_list = nltk.collocations.BigramCollocationFinder.from_documents([text for text in data[column]])
        if len(total_list.ngram_fd.items())>frequency:
            total_list.apply_freq_filter(frequency)        
        scores = total_list.score_ngrams(bigram_collocations.pmi)
    elif n == 3:
        trigram_collocations = nltk.collocations.TrigramAssocMeasures()
        total_list = nltk.collocations.TrigramCollocationFinder.from_documents([text for text in data[column]])
        if len(total_list.ngram_fd.items())>frequency:
            total_list.apply_freq_filter(frequency) 
        scores = total_list.score_ngrams(trigram_collocations.pmi)
    
    ngram_df = pd.DataFrame(scores, columns = ['ngram', 'pmi'])
    ngram_df = ngram_df.sort_values(by = 'pmi', axis = 0, ascending = False)
    to_keep = []

    if ngram_df.shape[0] == 0:
        return [], []
    
    for i in range(len(ngram_df)):
        ngram_val = ngram_df.loc[i, 'ngram']
        # print(ngram_val)
        tagged_ngram = nltk.pos_tag(ngram_val)
        ngram_df.loc[i, 'pos'] = str(tagged_ngram)
        # print(tagged_ngram)
        if n == 2:
            if (tagged_ngram[0][1] not in ['NN', 'JJ', 'VBG'] and tagged_ngram[1][1] not in ['NN']) or tagged_ngram[0][0] in stopwords_list or tagged_ngram[1][0] in stopwords_list or 'n' in ngram_val or 't' in ngram_val:
                to_keep.append(False)
            else:
                to_keep.append(True)
        elif n == 3:
            if (tagged_ngram[0][1] not in ['NN', 'JJ'] and tagged_ngram[2][1] not in ['NN']) or tagged_ngram[0][0] in stopwords_list or tagged_ngram[1][0] in stopwords_list or tagged_ngram[2][0] in stopwords_list or 'n' in ngram_val or 't' in ngram_val:
                to_keep.append(False)
            else:
                to_keep.append(True)
                
    tagged_list = ngram_df[to_keep][ngram_df['pmi'] > min_pmi]
    final_list = tagged_list['ngram']
    tagged_list.reset_index(inplace = True)
                     
    return final_list[:min(len(tagged_list),best_number)], tagged_list


def question_count(data, speaker_list = None, special_characters_rm = regular_expression):
    assert type(data) == type([]) or type(data) == type('Document'), "Data is not a list or string"
    assert type(speaker_list) == type([]) or speaker_list == None, "Speakers are not in a list"
    assert type(special_characters_rm) == re.Pattern, "Your special characters are not a regex Pattern object"
    assert len(data) == len(speaker_list), "Mismatch in data and speakers"

    status = ''
    is_valid_question = False
    qa_indices = []
    ans_indices = []
    questions = []
    answers = []

    for turn in range(len(data)):

        current_speaker = speaker_list[turn]
        print("\nCurrent speaker:", current_speaker)
        if 'host' in current_speaker.lower() or 'cnn' in current_speaker.lower():
            status = 'questioner'
        else:
            status = 'response'
        print("Status:", status)
        sentences = sent_tokenize(data[turn])

        for sent in sentences:
            if sent[-1] == '?' and status=='questioner' and not is_valid_question:
                is_valid_question = True
            
            elif (status == 'questioner' and is_valid_question) or (status == 'questioner' and turn == 0):
                is_valid_question = False

            if is_valid_question and status == 'questioner':
                print("Valid question:", sent)
                questions.append(sent)
                qa_indices.append(turn)
                break
        print(is_valid_question)
        if status == 'response' and is_valid_question:
            print("Answer:", data[turn])
            answers.append(data[turn])
            ans_indices.append(turn)
            is_valid_question = False

    return questions, answers, qa_indices, ans_indices
        

def generate_report(data):

    channel = data['id'].split('-')[0]
    program = data['program']
    utterances = data['utt']
    speakers = data['speaker']
    summary = data['summary']

    num_turns = turn_count(utterances)
    num_words_summary = word_count(summary)
    num_words_dialog = word_count(utterances)
    num_speakers = speaker_count(speakers)
    questions, answers, _, _ = question_count(utterances, speakers)
    num_qa_pairs = len(questions)
    question_lengths = [word_count(question) for question in questions]
    answer_lengths = [word_count(answer) for answer in answers]
    avg_q_length = 0
    avg_a_length = 0
    try:
        avg_q_length = np.sum(np.array(question_lengths))/num_qa_pairs
        avg_a_length = np.sum(np.array(answer_lengths))/num_qa_pairs
    except:
        print("No answers or questions")
    qa_pairs = {}
    if len(questions) == len(answers):  
        for question in range(len(questions)):
            qa_pairs[questions[question]] =  answers[question]

    return {'channel': channel, 'program': program, 'turns':num_turns, 'words_summary':num_words_summary, 'words_dialog':num_words_dialog, 'speakers':num_speakers,
    'avg_q':avg_q_length, 'avg_a':avg_a_length, 'num_q':num_qa_pairs}, qa_pairs




            



        



