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

@app.route('/analysis', methods = ["POST"])
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

        for i, row in data.iterrows():
            results[i] = preprocess.generate_report(row)
            ctm_transcript, _ = topic_model.ctm_model(row)
            word_images = topic_model.generate_word_cloud(row, ctm_transcript)
            topic_model_results[i] = [os.path.join(app.config['UPLOAD_FOLDER'], url) for url in word_images]
            print(word_images)

        return render_template("report.html", result_data = results, topic_data = topic_model_results)

@app.route('/locator', methods = ['POST'])
def locate():
    if request.method == 'POST':
        query = request.form['query']
        # file1 = open('models/sem_sim_gaussian_nb.pickle', 'rb')
        # model1 = pickle.load(file1)
        booster_model = xgb.Booster({'nthread': 4})  # init model
        booster_model.load_model('models\\xgbooster.model')
        w2vmodel = FastText.load('models\\fasttextmodel.model')
        db, other = dbp.retrieve_db()

        preprocessed_query = dbp.preprocess_data(query)
        temp_vector = pd.DataFrame()
    
        for word in preprocessed_query:
            embedding = w2vmodel.wv[word]
            temp_vector = temp_vector.append(pd.Series(embedding), ignore_index = True)
        current_vector = pd.DataFrame(temp_vector.mean()).transpose()

        query_df = pd.concat([current_vector]*len(other), ignore_index = True)
        query_df.columns = ['query_'+ str(col) for col in query_df.columns]

        new_index = np.max(db.post_id) + 1

        query_id = np.array([new_index]*len(db))

        db['qid2'] = query_id

        counts = pd.Series(db['qid1'].tolist() + db['qid2'].tolist()).value_counts()

        db['shared_words'] = db.apply(dbp.find_common_words, axis = 1)
        db['total_words'] =  db.apply(dbp.find_total_words, axis = 1)
        to_drop = db[db['total_words'] == 0].index
        db = db.drop(to_drop, axis = 0)
        db.reset_index(inplace = True)
        db['shared_ratio'] = db.apply(dbp.find_shared_ratio, axis = 1)
        db['countq1'] = db['qid1'].apply(lambda x: counts[x])
        db['countq2'] = db['qid2'].apply(lambda x: counts[x])

        q1 = pd.DataFrame(other)
        q2 = pd.DataFrame(query_df)
        q1.fillna(-9999, inplace = True)
        q2.fillna(-9999, inplace = True)

        numeric_features = db.loc[:, ['shared_ratio','countq1', 'countq2', 'qid1', 'qid2', 'total_words']]

        X = pd.concat((q1, q2, numeric_features), axis = 1)
        print(X.head())
        test_data = xgb.DMatrix(X)
        ypred = np.round(booster_model.predict(test_data))

        db['is_similar'] = ypred

        print("Number of relevant questions:", np.sum(db.is_similar))

        similar_questions = db[db['is_similar'] == 1]
        similar_questions['cosine_similarity'] = db.apply(dbp.get_cosine_simlarity, axis = 1)
        sorted_vals = similar_questions.sort_values(by = 'cosine_similarity', ascending = False)
        print(sorted_vals.head())
        return render_template('similarity_report.html', relevant = sorted_vals.head(5)['body_text'].tolist())


if __name__ == "__main__":
    app.run(debug=True)