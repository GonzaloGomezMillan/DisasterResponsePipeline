import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
        Args:
            messages_filepath (str): Path where the "messages.csv" is stored
            categories_filepath (str): Path where "categories.csv" is stored
        Return:
            df (DataFrame): Merged dataframe from messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''
    Function to clean the df already loaded and merged in the function "load_data".
    
    Args:
        df (DataFrame): Dataframe to clean, already loaded through the "load_data" function.
        
    Return:
        df (DataFrame): Dataframe already cleaned through the applicattion of the different steps of this function
    '''
    # create a dataframe of the 36 individual category columns
    categories_id = pd.DataFrame(df['id'])
    categories = df['categories'].str.split(pat=';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int').replace(2,1)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    # categories['id'] = categories_id
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)   
    
    return df


def save_data(df, database_filename):
    '''
    Function to save the DataFrame already loaded and cleaned from previous functions in a SQL database.
    
    Args:
        df (DataFrame): DataFrame to save in the database.
        database_filename: Name of the database in which the DataFrame has to be saved
    
    Return:
        None
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('ETLPipeline_Udacity', engine, index=False, if_exists = 'replace')
    return 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()