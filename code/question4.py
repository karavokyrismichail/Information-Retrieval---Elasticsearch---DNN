import pandas as pd
from gensim.models import word2vec
import numpy as np
import string
from sklearn.cluster import KMeans
from sklearn import preprocessing


#get the dataframes of csv files
users_df = pd.read_csv("BX-Users.csv")
books_df = pd.read_csv("BX-Books.csv")
ratings_df = pd.read_csv("BX-Book-Ratings.csv")


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



#function to get all book summaries
def get_book_summarires():
    list_of_summaries = []
    for i in range(len(books_df)):
        list_of_summaries.append(books_df.loc[i, "summary"])
    
    return list_of_summaries

def get_book_isbns():
    list_of_isbns = []
    for i in range(len(books_df)):
        list_of_isbns.append(books_df.loc[i, "isbn"])
    
    return list_of_isbns

# def get_book_caregory():
#     list_of_categories = []
#     for i in range(len(books_df)):
#         list_of_categories.append(books_df.loc[i, "category"])
    
#     return list_of_categories


# def unique(list1):
#     x = np.array(list1)
#     return (np.unique(x))


#antistoixia stis listes
book_isbns = get_book_isbns()
summaries = get_book_summarires()
summaries_vectors = calculate_vectors(summaries)
# book_categories = get_book_caregory()

# unique_categories = unique(book_categories)
# print("unique_categories   ", len(unique_categories)) 


summary_vectors_dict = dict(zip(book_isbns, summaries_vectors))
isbns_summary_vectors_df = pd.DataFrame.from_dict(summary_vectors_dict).fillna(0).T

X = isbns_summary_vectors_df
X_Norm = preprocessing.normalize(X)

kmeans = KMeans(n_clusters=8, random_state=42).fit(X_Norm)
labels_list = list(kmeans.labels_)

books_labels_df = pd.DataFrame({'isbn': book_isbns, 'cluster': labels_list})

merged_df = ratings_df.merge(books_labels_df, on="isbn")
merged_df = merged_df.merge(users_df, on="uid")


final_cut = (merged_df[~merged_df['age'].isnull()]).reset_index().drop(columns=['index', 'uid', 'isbn'])
print(final_cut)

print("\n examples: \n")
by_age = final_cut.query("age > 18 & cluster == 3 & rating > 5").shape[0]
print("number of people that are 18 years old or older and have rate the cluster 3 with 6+ \n", by_age)
by_age = final_cut.query("age > 25 & cluster == 7 & rating < 5 & location == 'nyc, new york, usa'").shape[0]
print("number of people that are located in nyc, new york, usa and are 25 years old or older and have rate the cluster 7 with 4- \n", by_age)