# Information-Retrieval

## Files
Name | Description
--- | ---
main.py | Information Retrieval System
transcript.pdf | A transcript of the runs on 3 test cases
requirements.txt | Project dependencies
README.pdf | Project description

## Run
```bash
$ python3 main.py <google_api_key> <google_engine_id> <precision> <query>
```

Credential | Key
--- | ---
Google Api Key | 
Google Engine Id | 

## Project Design
The system takes a query, retrieves search results from Google, collects user feedback, augments the query using the Rocchio algorithm, sorts the augmented query based on bigram score, and retrieves search results again using the new query.
1. ``get_parameters():``This function gets the input parameters from the user and prints them.
2. ``google_search():``This function retrieves search results from Google using the Google Api Key and Google Engine Id.
3. ``collect_feedback():``This function collects user feedback for the retrieved search results and divides the relevant and non-relevant documents into two lists. Also, the function records the number of relevant and non-relevant documents, which is used to calculate precision. (only html files are recorded)
4. ``augment_query():``This function uses the Rocchio algorithm to augment the query with two new words based on the relevant and non-relevant document lists. It returns the new query and the two words to be added to the query. The external libraries of ``sklearn.feature_extraction.text`` is used here, which function is introduced in the next part.
5. ``sort_query():``This function sorts the new query based on bigram score and returns the sorted query.

## Query-Modification Method
1. Augment new words: We use the Rocchio algorithm. 
  - First, we create an instance of the TfidfVectorizer class from scikit-learn using ``TfidfVectorizer(analyzer='word', stop_words='english')``. This class is used to convert related documents, irrelevant documents and old query into TF-IDF feature matrix.
  - Second, We compute the centroids of relevant and irrelevant documents according to the TF-IDF feature matrix.(``relevant_matrix.mean(axis=0)``,``non_relevant_matrix.mean(axis=0)``)
  - Third, We calculate the new query vector using the Rocchio algorithm. (``new_query_vector = alpha * query_vector + beta * relevant_centroid - gamma * non_relevant_centroid``) We set alpha = 1, beta = 0.75 and gamma = 0.15.
  - Fourth, We select two words with the lowest weight from new query vector as the new words. These two words are not duplicated with the old query.
  
1. Sort new query: We use N-gram model
  - First, we take the first word in new query as ``sorted_words``.
  - Second, We traverse the remaining words in the new query and insert the remaining word into ``sorted_words`` at the position with max bigram score.
  - The bigram score is calculated based on the number of occurrences of the bigram in the relevant document.

## Additional Information
We use "fileFormat" to ignore non-html files, but non-html files will still be displayed to the user.
