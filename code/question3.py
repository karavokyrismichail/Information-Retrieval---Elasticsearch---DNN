from elasticsearch import Elasticsearch
from question1 import search_user_input
import pandas as pd
import nltk
from gensim.models import word2vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
import numpy as np
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Softmax
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

es = Elasticsearch()

CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


#get the dataframes of csv files
ratings_df = pd.read_csv("BX-Book-Ratings.csv")
books_df = pd.read_csv("BX-Books.csv")

merged_df = ratings_df.merge(books_df, on="isbn")

'''
#first approach
#function for calculating vectors (word2vec) for every summary
def calculate_vectors(summaries):
    cleared_summaries_list = []
    letters_list = []
    vectors_list = []

    for summary in summaries:
        #clear the summaries, keep only improtant info
        summary = re.sub(r'[^\w\s]', '', summary)
        #print('re summary list: ', summary)
        tokenized_summary = word_tokenize(summary)
        #print('tokenized_summary: ', tokenized_summary)
        cleared_summary = ' '.join([word for word in tokenized_summary if not word in stopwords.words()])
        cleared_summaries_list.append(cleared_summary)
        #print('cleard summary list: ', cleared_summaries_list)
    model = word2vec.Word2Vec(cleared_summaries_list, min_count = 1, vector_size = 100, window = 5)

    for clear_summary in cleared_summaries_list:
        for letter in clear_summary:
            if letter != ' ':
                #print(letter)
                letter_matrix = model.wv[letter]
                #print("letter matrix: ", letter, letter_matrix)
                letters_list.append(letter_matrix)
        summary_vectror = sum(letters_list) / len(letters_list)
        #print("summary_vectror: ", summary_vectror)
        letters_list.clear()
        vectors_list.append(summary_vectror)

    #print(vectors_list)
    #word_vectors = model.wv['b']
    #print(word_vectors)
    return vectors_list
'''

def remove_digits(s):
    s = ''.join([i for i in s if not i.isdigit()])
    return s
#vectorizing the summaries

def calculate_vectors(summaries):

    summary_vectors = []
    for i in range(len(summaries)):
        summaries[i] = remove_digits(summaries[i])

    tokenized_summaries = [summary.translate(str.maketrans('', '', string.punctuation)).split() for summary in summaries]
    model = word2vec.Word2Vec(tokenized_summaries, vector_size=100, min_count=1)
    word_vectors = model.wv
    #calculating total vector for every summary
    shape = np.array(word_vectors.get_vector(tokenized_summaries[i][0])).shape
    for i in range(len(tokenized_summaries)):
        temp_vector_1 = np.zeros(shape)
        for word in tokenized_summaries[i]:
            if word != "":
                temp_vector_2 = np.array(word_vectors.get_vector(word))
                temp_vector_1 += temp_vector_2
        summary_vectors.append(temp_vector_1 / len(tokenized_summaries[i]))

    return summary_vectors


#function for getting the summaries and the ratings of the books that the user has rate.
def get_users_ratings(userid):
    #max_uid = merged_df['uid'].value_counts().idxmax()
    rat_list = []
    sum_list = []
    uid_summaries_df = (merged_df.loc[merged_df['uid'] == userid]).reset_index()
    for i in range(len(uid_summaries_df)):
        sum_list.append(uid_summaries_df.loc[i, "summary"])
        rat_list.append(uid_summaries_df.loc[i, "rating"])

    return sum_list, rat_list

'''
def get_average_book_ratings ():
    # remove any ratings that are zero
    remove_zero_ratings_df = ratings_df[ratings_df.rating != 0]
    average_ratings_df = remove_zero_ratings_df.groupby('isbn').mean()
    # print(average_ratings_df)
    merged_df = books_df.merge(average_ratings_df, on="isbn")

    return merged_df
'''

def getPredictionLabels(predictions):
    # This function calculates the real label for each prediction.
    # Returns the index of the biggest probability
    return [np.argmax(prediction) for prediction in predictions]



# average_ratings = get_average_book_ratings()

def neural_network_prediction(usersid):

    summaries_list, ratings_list = get_users_ratings(usersid)
    #print(ratings_list)
    vec_list = calculate_vectors(summaries_list)

    X = vec_list
    y = ratings_list #[(rating*2)-1 for rating in ratings_list]

    #train, split values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = Sequential()
    model.add(Dense(32, input_dim=len(X_train[0]), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(11))

    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(np.array(X_train), np.array(y_train), epochs=10, verbose=0)

    probability_model = Sequential([model, Softmax()])

    predictions = model.predict(np.array(X_test))
    y_pred = predictions #[(prediction+1)/2 for prediction in predictions]
    
    # print("\n-------------------------------Scores-------------------------------\n")
    # print("Precision: " + str(precision_score(y_test, y_pred, labels=LABELS, average='micro')))
    # print("Recall: " + str(recall_score(y_test, y_pred, labels=LABELS, average='micro')))
    # print("F1 Score: " + str(f1_score(y_test, y_pred, labels=LABELS, average='micro')))
    # print("\n--------------------------------------------------------------------\n")
    #print(classification_report(y_test, y_pred, target_names=CLASSES))

    #predictions = model.predict(np.array(not_rated))
    #y_pred = predictions #[(prediction+1)/2 for prediction in predictions]
    return probability_model


#querry for books
unfiltered_results = search_user_input ()

#user's id
user_id_input = int(input('Please give your id: '))

need_ratings = []
for index, result in enumerate(unfiltered_results['hits']['hits']):

    isbn = result['_source']['isbn']

    #get list of the books that have not be rated from the user
    user_rating = float(merged_df.loc[(merged_df['isbn'] == isbn) & (merged_df['uid'] == user_id_input)].iloc[0]['rating']) if ((merged_df['isbn'] == isbn) & (merged_df['uid'] == user_id_input)).any() else False
    if user_rating == False:
        need_ratings.append(isbn)

#get the summaries of the books that have not be rated from the user
summaries_not_rated = []
for book in need_ratings:
    not_rated_df = (books_df.loc[books_df['isbn'] == book]).reset_index()
    for i in range(len(not_rated_df)):
        summaries_not_rated.append(not_rated_df.loc[i, "summary"])

not_rated_summaries_vectors = calculate_vectors(summaries_not_rated)

prediction_machine = neural_network_prediction(user_id_input)

user_predicted_ratings_list = []
for item in not_rated_summaries_vectors:
    new_ratings_predicted = prediction_machine.predict(np.array([item]))
    user_predicted_ratings_list.append(new_ratings_predicted)

predicted_label = getPredictionLabels(user_predicted_ratings_list)


print(predicted_label)

#get dict with the new ratings and isbns
new_ratings_dict = dict(zip(need_ratings, predicted_label))

results = []
for index, result in enumerate(unfiltered_results['hits']['hits']):

    title = result['_source']['book_title']
    isbn = result['_source']['isbn']
    bm25_rating = result['_score']

    if isbn in need_ratings:
        final_score = 0.5*bm25_rating + 0.5*new_ratings_dict[isbn]

    else:
        user_rating = float(merged_df.loc[(merged_df['isbn'] == isbn) & (merged_df['uid'] == user_id_input)].iloc[0]['rating'])
        final_score = 0.5*bm25_rating + 0.5*user_rating

    results.append([result['_source']['book_title'], final_score])


print("\n Preference --- Book Name --- Personalised Score \n", sorted(results, key=lambda x: x[1], reverse=True))