from gensim.models import Word2Vec, FastText
import numpy as np
import pandas as pd
import sys
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import preprocess as preprocess
import topic_model as topic_model
import databasepreprocess as dbp
import pickle
import xgboost as xgb
import os

app = Flask(__name__)
app.debug = True

IMAGE_FOLDER = os.path.join('outputs')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcriptpage')
def transcriptpage():
    return render_template('transcripts.html')

@app.route('/simpage')
def simpage():
    return render_template('similarities.html')

@app.route('/analysis', methods = ["POST", "GET"])
def analyse():
    if request.method == "POST":
        if "fileUploaded" not in request.files:
            flash("No file")
            redirect(request.url)

        file = request.files["fileUploaded"]
        data = pd.DataFrame(pd.read_json(file))
        data.reset_index(inplace = True)
        results = {}
        topic_model_results = {}
        qa_pairs_list = {}
        for i, row in data.iterrows(): 
            results[i], qa_dict = preprocess.generate_report(row)
            qa_pairs_list[i] = qa_dict
            ctm_transcript, _ = topic_model.ctm_model(row)
            word_images = topic_model.generate_word_cloud(row, ctm_transcript)
            topic_model_results[i] = [os.path.join(app.config['UPLOAD_FOLDER'], url) for url in word_images]
            print(word_images)
            print(qa_dict.items())

        return render_template("report.html", result_data = results, topic_data = topic_model_results, question_pairs = qa_dict)

@app.route('/locator', methods = ['POST'])
def locate():
    if request.method == 'POST':
        query = request.form['query']
        print("Query is:", query)
        # file1 = open('models/sem_sim_gaussian_nb.pickle', 'rb')
        # model1 = pickle.load(file1)
        booster_model = xgb.Booster({'nthread': 4})  # init model
        booster_model.load_model('models\\xgbooster.model')
        w2vmodel = FastText.load('models\\fasttextmodel.model')
        print("All models loaded.")
        db, encoded_db, other = dbp.retrieve_db()
        db['preprocessed_q1'] = db['preprocessed_q1'].apply(lambda x: x.strip('][').split(','))
        db['preprocessed_q1'] = db['preprocessed_q1'].apply(lambda x: [word.strip(" ' ") for word in x])
        print("Retrieved database.")

        preprocessed_query = dbp.preprocess_data(query)
        print("Preprocessed:", preprocessed_query)
        temp_vector = pd.DataFrame()
    
        for word in preprocessed_query:
            embedding = w2vmodel.wv[word]
            temp_vector = temp_vector.append(pd.Series(embedding), ignore_index = True)
        current_vector = pd.DataFrame(temp_vector.mean()).transpose()
        print("Encoded query")
        query_preprocessed_df = pd.DataFrame(columns = ['preprocessed_query'])
        query_preprocessed_df['preprocessed_query'] = [preprocessed_query]*len(db)
        db = pd.concat([db, query_preprocessed_df], axis = 1)

        query_df = pd.concat([current_vector]*len(encoded_db), ignore_index = True)
        query_df.columns = ['query_'+ str(col) for col in query_df.columns]
        # print("Query as a dataframe:", query_df.head())

        new_index = np.max(db.post_id) + 1
        print("New index:", new_index)

        query_id = [new_index]*len(db)

        db['qid2'] = query_id

        counts = pd.Series(db['qid1'].tolist() + db['qid2'].tolist()).value_counts()

        db['shared_words'] = db.apply(dbp.find_common_words,  col1 = 'preprocessed_q1', col2 = 'preprocessed_query', axis = 1)
        db['total_words'] =  db.apply(dbp.find_total_words, col1 = 'preprocessed_q1', col2 = 'preprocessed_query', axis = 1)
        to_drop = db[db['total_words'] == 0].index
        db = db.drop(to_drop, axis = 0)
        db.reset_index(inplace = True)
        db['shared_ratio'] = db.apply(dbp.find_shared_ratio, axis = 1)
        db['countq1'] = db['qid1'].apply(lambda x: counts[x])
        db['countq2'] = db['qid2'].apply(lambda x: counts[x])
        print("Built features")
        print(db.loc[15566, ['shared_words', 'preprocessed_q1', 'preprocessed_query']])
        print(type(db.loc[0, 'preprocessed_q1']))
        print(type(db.loc[0, 'preprocessed_query']))
        print(type(db.loc[0, 'preprocessed_q1'][0]))
        print(type(db.loc[0, 'preprocessed_query'][0]))

        q1 = pd.DataFrame(encoded_db)
        q2 = pd.DataFrame(query_df)
        q1.fillna(-9999, inplace = True)
        q2.fillna(-9999, inplace = True)
        print("Filled nulls")

        numeric_features = db.loc[:, ['shared_ratio','countq1', 'countq2', 'qid1', 'qid2', 'total_words']]

        X = pd.concat((q1, q2, numeric_features), axis = 1)
        print("Full dataset:\n", X.head(), X.columns)
        X.drop('Unnamed: 0', axis = 1, inplace = True)
        test_data = xgb.DMatrix(X)
        ypred = np.round(booster_model.predict(test_data))

        db['is_similar'] = ypred

        print("Number of relevant questions:", np.sum(db.is_similar))
        similar_questions = db[db['is_similar'] == 1]
        # similar_questions.drop_duplicates(keep = 'first', inplace = True)
        relevant_encodings = []
        if np.sum(db.is_similar) == 0:
            similar_questions = db
            relevant_encodings = X.iloc[:, 0:64]
            print("Relevant encodings:", relevant_encodings.head())
        else:
            relevant_encodings = X.iloc[pd.Series(similar_questions.index.values), 0:64]
            print("Relevant encodings:", relevant_encodings.head())
        print("Original Similar questions:", similar_questions.head().loc[:, ['shared_words', 'shared_ratio']])

        similar_questions = pd.concat([relevant_encodings,similar_questions], axis = 1)
        similar_questions_new = similar_questions[(similar_questions['shared_ratio'] > 0) | (similar_questions['shared_words'].str.len() > 0)]
        if similar_questions_new.shape[0] != 0:
            similar_questions = similar_questions_new
        similar_questions['cosine_similarity'] = similar_questions.apply(dbp.get_cosine_simlarity, axis = 1)
        print(similar_questions.head())
        similar_questions.drop_duplicates(['cosine_similarity', 'shared_ratio', 'total_words'], inplace = True, keep = 'first')
        sorted_vals = similar_questions.sort_values(by = 'cosine_similarity', ascending = False)
        print(sorted_vals.head())
        return render_template('similarity_report.html', relevant = sorted_vals.head()['body_text'].tolist())


if __name__ == "__main__":
    app.run(debug=True)