import sys

import nltk
import pandas as pd
import sklearn
from flask import Flask, request
from flask.templating import render_template

from nlp.exception import SpamException
from nlp.nlp import SpamClassifier, SpamTransformer

app = Flask(__name__)


@app.route('/',methods=['GET'])
def homepage(): 
    return render_template('index.html')

@app.route('/review',methods=['GET','POST'])
def index(): 
    if request.method == 'POST': 
        try : 
            
            text = request.form['content']
            dataset_path = r"C:\Users\manev\Desktop\iNeuron\NLP\Spam classification\smsspamcollection\SMSSpamCollection"
            corpus = SpamClassifier(dataset_path=dataset_path).get_corpus()
            dataset = SpamClassifier(dataset_path=dataset_path).get_dataset()
            spam = SpamTransformer(corpus=corpus,dataset=dataset)
            spam.feature_transform()
            y_pred = spam.predict(x=[text])
            if y_pred == 0: 
                result = "ham"
            elif y_pred == 1 : 
                result = "Spam"

            dict1 = {"Text":text , "Prediction":result}

            reviews = [dict1]
            
            return render_template('results.html',reviews=reviews)

        except Exception as e : 
            raise SpamException(e,sys) from e 
    
    else : 
        render_template('index.html')







if __name__ == "__main__": 
    app.run(debug=True)