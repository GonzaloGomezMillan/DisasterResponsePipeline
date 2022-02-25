import sys

from sqlalchemy import create_engine
import sqlite3
import pandas as pd
import os
import nltk
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report

def load_data(database_filepath):
    '''
    Function which read a SQL database
    
    Args:
        database_filepath (str): File path to the database to load.
        
    Return:
        X
        Y
        category_names (list): Names of the category variables that will be considered for the model
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM ETLPipeline_Udacity", engine)
    X = df.message.values
    category_names = df.columns.drop(['id','message','original','genre','related','request'])
    df = df.fillna(0)
    Y = df[category_names].astype('int').values
    return X, Y, category_names


def tokenize(text):
    '''
    Function which tokenize a determined text.
    
    Args:
        text (str): Message to be tokenized
    
    Return:
        tokens (list): List of the processed tokens
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ",text)
    tokens = word_tokenize(text.lower())
    tokens = [tok for tok in tokens if tok not in stopwords.words('english')]
    tokens = [WordNetLemmatizer().lemmatize(tok) for tok in tokens]
    return tokens


def build_model():
    '''
    Function in charge of building the model through a pipeline.
    
    Args:
        None
    
    Return:
        model: Model created through a pipeline
    '''
    pipeline = Pipeline([
                    ('vect',CountVectorizer(tokenizer = tokenize)),
                    ('tfidf',TfidfTransformer()),
                    ('clf',MultiOutputClassifier(RandomForestClassifier()))
                    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    pipeline.get_params()
    
    parameters = {
            'clf__estimator__n_estimators': [5,10]
            }

    cv = GridSearchCV(pipeline, param_grid = parameters)
    cv.fit(X_train,y_train)
    
    model = Pipeline([
                    ('vect',CountVectorizer(tokenizer = tokenize)),
                    ('tfidf',TfidfTransformer()),
                    ('clf',MultiOutputClassifier(RandomForestClassifier(n_estimators= 10)))
                    ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function elaborated to evaluate the model created in the "build model" function
    
    Args:
        model: Object created through the model created in the "build model" function
        X_test: X test matrix used to evaluate the model
        Y_test: Y test matrix used to evaluate the model
        category_names (list): list with the category names that are part of the commented model
    
    Return 
        None
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    return


def save_model(model, model_filepath):
    pickle.dump(model, open('model_disaster_response_pipelines.pkl','wb'))
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()