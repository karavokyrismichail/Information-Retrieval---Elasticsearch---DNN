# Information-Retrieval---Elasticsearch---DNN

Project for  Information Retrieval Course of the Department of Computer Engineering & Informatics.

## Description

A book search engine based on Elasticsearch that will decides the order of presentation of the results using machine learning techniques.

## Scripts

### insert_elastic

- Imports data (Books) from csv file to Elasticsearch using json.

### question1

- Given an alphanumeric input from user, it returns the list of Books that match it, arranged in descending order according to the default similarity metric of Elasticsearch.

### question2

- Given an alphanumeric input and a user ID, it returns the results in descending order according to:
1. The default similarity metric of Elasticsearch.
2. The user's rating on book (if available).
3. The average ratings from all the other users.

### question3

- In this query I tried to improve the quality of the classification by filling in the missing ratings.
- Given same input as before, it trains a sequential neural network model from excisting user's ratings to guess how this user would rate the rest of the Books.
- To train the model, I transformed the dataset by converting the book summaries in vectors utilizing the Word Embeddings technique.
- Finally it combines the results with the default similarity metric of Elasticsearch and the average ratings from all the other users to achieve the best ranking.

### question4

- Divides the books into clusters based on their plot
- To achieve that, I used the vectors of summaries that were created in the previous question to give them as input to k-means.
- The algorithm is modified to use as metric of distance the cosine similarity (two elements with high similarity must end in the same cluster).
- Finally, identifies correlations between the clusters created and how users rate based on their demographics.

## Tech stack
- Python, VSC, Elasticsearch, scikit-learn, Keras, Word2vec, NLTK, mumpy, pandas
