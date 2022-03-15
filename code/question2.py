from elasticsearch import Elasticsearch
from question1 import search_user_input
import pandas as pd

es = Elasticsearch()

ratings_df = pd.read_csv("BX-Book-Ratings.csv")
books_df = pd.read_csv("BX-Books.csv")

def get_average_book_ratings ():
    # remove any ratings that are zero
    remove_zero_ratings_df = ratings_df[ratings_df.rating != 0]

    # https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.GroupBy.mean.html
    average_ratings_df = remove_zero_ratings_df.groupby('isbn').mean()

    # print(average_ratings_df)
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html 
    merged_df = books_df.merge(average_ratings_df, on="isbn")

    return remove_zero_ratings_df, merged_df

nonzero_ratings_df, average_ratings = get_average_book_ratings()

# print(average_ratings)

# from the previous question get the score results
unfiltered_results = search_user_input ()

user_id_input = int(input('Please give your id: '))

results = []
for index, result in enumerate(unfiltered_results['hits']['hits']):

    title = result['_source']['book_title']
    isbn = result['_source']['isbn']
    bm25_rating = result['_score']

    # get the user rating for a specific isbn IF there is a row (in the nonzero_ratings df) that has both the user id that was given by the user and the isbn of the book in the unfiltered results loop above
    # in case that there is not a match (required user id and isbn in the same row), set the user_rating variable to false 
    user_rating = float(nonzero_ratings_df.loc[(nonzero_ratings_df['isbn'] == isbn) & (nonzero_ratings_df['uid'] == user_id_input)].iloc[0]['rating']) if ((nonzero_ratings_df['isbn'] == isbn) & (nonzero_ratings_df['uid'] == user_id_input)).any() else False

    # get the average book rating for a isbn if it exists in the dataframe
    # in case that there is not a match (required isbn and isbn in the same row), set the average_book_rating variable to false
    average_book_rating = float(average_ratings.loc[average_ratings['isbn'] == isbn].iloc[0]['rating']) if (average_ratings['isbn'] == isbn).any() else False
    
    '''
    UNCOMMENT TO DEMONSTRATE HOW IT WORKS IN DETAILS -- try flu, with uid=8
    print('\n\n\n')
    print(user_rating)
    print(average_book_rating)
    print(bm25_rating)
    print('\n\n\n')
    '''

    # In all cases below the BM25 score will exist

    # if both variables exist
    if average_book_rating and user_rating:
        final_score = 0.4*bm25_rating + 0.4*user_rating + 0.2*average_book_rating
        # print('1')

    # if only the average_book_rating exists
    elif average_book_rating and not user_rating:
        final_score = 0.6*bm25_rating + 0.4*average_book_rating
        # print('2')

    # if only the user_rating exists
    elif user_rating and not average_book_rating:
        final_score = 0.6*bm25_rating + 0.4*user_rating
        # print('3')
    
    # if none of the variables exist (consequently the score will be determined by the BM25 similarity)
    else:
        final_score = bm25_rating
        # print('4')

    results.append([result['_source']['book_title'], final_score])

print("\n Preference --- Book Name --- Personalised Score \n", sorted(results, key=lambda x: x[1], reverse=True))
