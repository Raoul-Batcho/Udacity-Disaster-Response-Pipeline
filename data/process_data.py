from sqlalchemy import create_engine
import sys
import pandas as pd

def load_data(messages_data_path, categories_data_path):
    """
    This function takes two arguments, two files paths and load the data into the respective Pandas Dataframes and then merges both of the Dataframes.
    It returns the Dataframe resulting from the merge function.
    """
    messages_df = pd.read_csv(messages_data_path)
    categories_df = pd.read_csv(categories_data_path)
    merged_df = pd.merge(messages_df, categories_df, on = 'id')
    return merged_df


def clean_data(merged_df):
    """
    This fuction takes as argument the merged dataframe and retrns the result after cleaning.
    """
    
    # dataframe of the thirty six individual category columns
    categories = merged_df['categories'].str.split(pat=';', expand = True)
    
    # look at the first row 
    row = categories.iloc[0]
    
    # extraction of a list of new column names for categories
    category_colnames = row.apply(lambda x:x[:-2])
    
    # renaming categories' columns
    categories.columns = category_colnames
    
    # converting category values only 0 or 1
    for col in categories:
       
        categories[col] = categories[col].astype(str).str[-1]
       
        categories[col] = categories[col].astype(int)
    
    # Remove original categories column from 'merged_df'
    merged_df = merged_df.drop('categories', axis = 1)
    
    # concatenation of the original dataframe with the new 'categories' dataframe
    merged_df = pd.concat([merged_df, categories], axis = 1)
    
    # Remove duplicates
    merged_df.drop_duplicates(inplace = True)
    
    # Remove 'child_alone' column 
    merged_df = merged_df.drop('child_alone', axis = 1)
    
    # Replace '2' with '1' in "related' column
    merged_df['related'] = merged_df['related'].map(lambda x: 1 if x==2 else x)
    
    cleaned_df = merged_df 
    return cleaned_df


def save_to_sql(cleaned_df, database_filename):
    """
    This function takes as argument, the cleaned dataframe and save cleaned data into sqlite database
    """
    engine = create_engine('sqlite:///'+ database_filename)
    cleaned_df.to_sql('DisasterResponse_table', engine, index = False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_data_path, categories_data_path, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_data_path}\n    CATEGORIES: {categories_data_path}')
              
        df = load_data(messages_data_path, categories_data_path)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_to_sql(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide messages data filepath, then categories '\
              'datasets filepath as the first and second argument respectively, as '\
              'then the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
