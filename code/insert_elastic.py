from elasticsearch import helpers, Elasticsearch
import pandas as pd
import json

es = Elasticsearch(HOST="http://localhost", PORT=9200)
es = Elasticsearch()

books_df = pd.read_csv("BX-Books.csv")

books_json = books_df.to_json(orient='records')

books_dict = json.loads(books_json)

es.indices.delete(index="books_index", ignore=[400, 404])

helpers.bulk(es, books_dict, index="books_index")