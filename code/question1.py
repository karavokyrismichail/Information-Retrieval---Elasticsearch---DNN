from elasticsearch import helpers, Elasticsearch

es = Elasticsearch()

# getting the user's input for the books title and returning the results based on BM25 similarity
def search_user_input ():
    book_title_input = input('Please give the name of the book title: ')
    body = {
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/sort-search-results.html
        "sort": [
            "_score"
        ],
        "query": {
            "match": {
                "book_title": book_title_input
            }
        }
    }
    
    res = es.search(index='books_index', body=body)
    return res
    
# filter and show the results 
def show_results ():

    unfiltered_results = search_user_input ()

    #print(unfiltered_results)
    
    print('Preference --- Book Name --- ISBN --- similarity Score')
    for index, result in enumerate(unfiltered_results['hits']['hits']):
        print(str(index+1) + '\t' + result['_source']['book_title'] + '\t' + result['_source']['isbn'] + '\t\t\t' + str(result['_score']))


if __name__ == "__main__":
    show_results ()
