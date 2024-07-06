import pandas as pd
import re
import sys
import os
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle


import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sqlalchemy import create_engine
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def load_data(database_file_path):
    """
    This fuction takes as argument, the data base file path and returns input and output variables
    """
    engine = create_engine('sqlite:///' + database_file_path)
    df = pd.read_sql_table('DisasterResponse_table', engine)
    
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    This function takes raw text data as argument and returns clean tokenized data
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def model_builder(clf = AdaBoostClassifier()):
    """
    This function takes a machine learning Classifier model as argument and returns a model pipeline after performing grid search
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(clf))
    ])
    
    parameters = {
        'clf__estimator__learning_rate':[0.01, 0.1, 1.0],
        'clf__estimator__n_estimators':[10,20,30]
    
    }
        
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=3) 
    
    return cv
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function takes as arguments the model - ML model, the test messages X_text, categories Y_test, category_names for y and returns
    the scores (precision, recall, f1-score) for each output category of the dataset.
    """
    Y_pred_test = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred_test, target_names=category_names))
    

def save_model(model, model_filepath):
    """
    This function saves the model as a pickle file
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = model_builder()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
