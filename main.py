import sys
import numpy as np
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer


# Global Variables
MAX_ITERATIONS = 5
GOOGLE_API_KEY = ""
GOOGLE_ENGINE_ID = ""
PRECISION = 0.0
QUERY = ""

def get_parameters():
    """
    Get parameters from the inputs, and store the value in global variables.
    """
    inputs = sys.argv
    global GOOGLE_API_KEY, GOOGLE_ENGINE_ID, PRECISION, QUERY
    # If the input format or value is incorrect, print the error message
    if len(sys.argv) != 5:
        print("Correct format: main.py <google api key> <google engine id> <precision> <query>")
    elif float(sys.argv[3]) < 0 or float(sys.argv[3]) > 1:
        print("The range of <precision> is 0 - 1")
    else:
        GOOGLE_API_KEY = inputs[1]
        GOOGLE_ENGINE_ID = inputs[2]
        PRECISION = float(inputs[3])
        QUERY = inputs[4]
        # Print parameters
        print("Parameters: ")
        print("Search Key = " + GOOGLE_API_KEY)
        print("Search Engine ID = " + GOOGLE_ENGINE_ID)
        print("Target Precision = " + str(PRECISION))
        print("QUERY = " + QUERY)
        print("Google Search Results:")


def google_search(**kwargs):
    """
    Get the result from Google Search Engine.
    """
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    res = service.cse().list(q=QUERY, cx=GOOGLE_ENGINE_ID, **kwargs).execute()
    result = res['items']
    return result

def collect_feedback():
    """
    Collect the feedback of the search results and store relevant and non-relevant results to two lists.
    """
    relevant_list = []
    non_relevant_list = []
    num_total, num_relevant, num_non_relevant = 0, 0, 0
    search_results = google_search()
    for item in search_results:
        num_total += 1
        print("Result " + str(num_total))
        print("[")
        print(" URL: " + item['link'])
        print(" Title: " + item['title'])
        print(" Summary: " + item['snippet'])
        print("]")
        
        feedback = input("Relevant(Y/N)? ")
        # Get the feedback from user.
        if feedback == 'Y' or feedback == 'y':
            if "fileFormat" not in item:
                num_relevant += 1
                relevant_list.append(item['title'] + item['snippet'])
        else:
            if "fileFormat" not in item:
                num_non_relevant += 1
                non_relevant_list.append(item['title'] + item['snippet'])

    return num_total, num_relevant, num_non_relevant, relevant_list, non_relevant_list


def augment_query(relevant_docs, non_relevant_docs, query, alpha = 1, beta = 0.75, gamma = 0.15):
    """
    Augment the query with two new words using Rocchio algorithm.
    """
    # Create TfidfVectorize object
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')

    # Vectorize the relevant documents and calculate the centroid
    relevant_matrix = vectorizer.fit_transform(relevant_docs)
    relevant_centroid = relevant_matrix.mean(axis=0)

    # Vectorize the non-relevant documents and calculate the centroid
    non_relevant_matrix = vectorizer.transform(non_relevant_docs)
    non_relevant_centroid = non_relevant_matrix.mean(axis=0)

    # Vectorize the current query
    query_vector = vectorizer.transform([query])

    # Calculate the new query vector using the Rocchio algorithm
    new_query_vector = alpha * query_vector + beta * relevant_centroid - gamma * non_relevant_centroid

    # Get the feature names and corresponding weights of the new query vector
    feature_names = vectorizer.get_feature_names_out()
    weights = new_query_vector.data
    weights = weights / np.linalg.norm(weights)

    # Sort the feature names by their corresponding weights in descending order
    sorted_indices = np.argsort(-weights)
    sorted_feature_names = [feature_names[idx] for idx in sorted_indices][0].tolist()
    
    # Find the two new words to add to the query
    expand_word = []
    for names in sorted_feature_names:
        if names not in query:
            expand_word.append(names)
        if len(expand_word) == 2:
            break

    # Combine the new words with the original query
    new_query = query + " " + expand_word[0] + " " + expand_word[1]

    return new_query, expand_word[0], expand_word[1]

def sort_query(new_query, relevant_docs):
    """
    Sort the new query based on bigram score.
    """
    # Initialize words and sorted words
    words = new_query.split()
    sorted_words = [words[0]]
    # Convert the relevant_docs to lowercase for matching
    relevant_docs = [x.lower() for x in relevant_docs]
    # Iterate through the sorted words list and insert the current word at the position with max bigram score
    for word in words[1:]:
        max_score = 0
        index = -1
        for i, sorted_word in enumerate(sorted_words):
            pair = (word, sorted_word)
            # Calculate the bigram score as the number of occurrences of the bigram in the document collection
            bigram_score = 0
            for document in relevant_docs:
                if ' '.join(pair) in document:
                    bigram_score += 1

            if max_score < bigram_score:
                max_score = bigram_score
                index = i
        # If the bigram has not appeared, append it at the end
        if index != -1:
            sorted_words.insert(index, word)
        else:
            sorted_words.append(word)

    return ' '.join(sorted_words)

def main():
    # Get the value of the input parameters.
    get_parameters()
    global QUERY
    for i in range(MAX_ITERATIONS):
        print("=====================")
        print(f'This is iteration #{i + 1}')

        results = google_search(num=10)

        # The results returned are not enough
        if len(results) < 10:
            print('The number of results are less the 10')
            break

        # Collect the feedback from users.
        num_total, num_relevant, num_non_relevant, relevant_list, non_relevant_list = collect_feedback()

        if num_relevant == 0:
            print('Terminate: No result is relevant in the first iteration')
            break

        # Calculate the current precision
        curr_precision = num_relevant / (num_relevant + num_non_relevant)
        print("=====================")
        print("FEEDBACK SUMMARY")
        print(f'Current Query : {QUERY}')

        # Check if desired precision is reached
        if curr_precision < PRECISION:
            print(f'Current Precision : {curr_precision}')
            print(f'Still below the desired precision of {PRECISION}')

            # Augment current query by adding two words
            new_query, expand_word_1, expand_word_2 = augment_query(relevant_list, non_relevant_list, QUERY)
            # Sort the new query based on bigram score
            new_query = sort_query(new_query, relevant_list)
            print("Indexing results ....")
            print("Indexing results ....")
            print(f'Using these two words to augment the query: {expand_word_1} {expand_word_2}')
            QUERY = new_query
            print("Current query: " + QUERY)
        else:
            print(f'Current Precision : {curr_precision}')
            print("Reached the target precision.")
            break


if __name__ == '__main__':
    main()
