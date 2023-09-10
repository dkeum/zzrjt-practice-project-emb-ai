# import libraries
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import requests
import json

df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')
df = df[0:4]
df['reviewText'][3] = "I hate this game"
# print(df['reviewText'][3])

analyzer = SentimentIntensityAnalyzer()


# create preprocess_text function
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# create get_sentiment function
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment

def sentiment_analyzer(text_to_analyse):
   # url = 'https://sn-watson-sentiment-bert.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict'
   # myobj = { "raw_document": { "text": text_to_analyse } }
   # header = {"grpc-metadata-mm-model-id": "sentiment_aggregated-bert-workflow_lang_multi_stock"}
   # response = requests.post(url, json = myobj, headers=header)
   # formatted_response = json.loads(response.text)
   # label = formatted_response['documentSentiment']['label']
   # score = formatted_response['documentSentiment']['score']

   label = analyzer.polarity_scores(text_to_analyse)
   highest_prob = max(label['neg'],label['neu'],label['pos'])
   rv_sentiment = ""
   for key,value in label.items():
      # print("This is the key: " + key + " This is the value: " + str(value))
      if highest_prob == value and key == "neg":
         rv_sentiment = "SENT_NEGATIVE"
         break
      elif highest_prob == value and key == "neu":
         rv_sentiment = "SENT_NEUTRAL"
         break
      elif highest_prob == value and key == "pos":  
         rv_sentiment = "SENT_POSITIVE"
         break
        
   return {'label': rv_sentiment, 'score': analyzer.polarity_scores(text_to_analyse)}

df['reviewText'] = df['reviewText'].apply(preprocess_text)
df['sentiment'] = df['reviewText'].apply(get_sentiment)

# print(df)
# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(df['Positive'], df['sentiment']))

# from sklearn.metrics import classification_report
# print(classification_report(df['Positive'], df['sentiment']))

# print(sentiment_analyzer("I am neutral on Python"))