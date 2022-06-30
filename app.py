from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from flask import Flask, request, render_template
from nltk.tokenize import word_tokenize
from keras.models import load_model
import numpy as np
import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

app = Flask(__name__)

model = load_model('Depression_LSTM_output.h5')


@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict", methods=['POST'])
def predict():
    message = request.form["message"]

    message = [message]

    def clean_text(data):
        data = re.sub(r'http\S+', '', data)
        data = re.sub('[^a-zA-Z]', ' ', data)
        data = data.lower()

        return data

    corpus = []
    for i in range(0, len(message)):
        data = re.sub('[^a-zA-Z]', ' ', message[i])
        data = data.lower()
        data = data.split()

        data = [ps.stem(word) for word in data if not word in stopwords.words('english')]
        data = ' '.join(data)
        corpus.append(data)

    voc_size = 10000
    onehot_repr_short = [one_hot(words, voc_size) for words in corpus]

    sent_len = 5850
    embedded_docs_msg = pad_sequences(onehot_repr_short, padding='pre', maxlen=sent_len)

    pred = model.predict(embedded_docs_msg)

    class_names = ['non-suicide','suicide']

    if model.predict([embedded_docs_msg]) > 0.5:
        my_pred = class_names[0]
    else:
        my_pred = class_names[1]
      
    return render_template('Article.html', data=my_pred)


if __name__ == "__main__":
    app.run(debug=True)
