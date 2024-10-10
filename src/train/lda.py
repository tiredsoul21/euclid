""" Latent Dirichlet Allocation (LDA) model training using Gibbs Sampling. """
import json
import random
import time

from scipy.special import gammaln
import matplotlib.pyplot as plt

import numpy as np

STOP_WORDS = set(['peopl', 'us', 'person', 'he', 'she', 'him', 'his', 'her', 'parent', 'back', 'those', 'these', 'cant'])

# Reddit data (Not used)
DATA_PATH = '/home/derrick/data/reddit/teachers/Teachers.json'
# The list of words as keys.  Should have the global word usage pery word
WORD_LIST_PATH = '/home/derrick/data/reddit/teachers/wordlist.json'
# The key(id) value(text) pair of the documents
DOC_PATH = '/home/derrick/data/reddit/teachers/documents.json'

# LDA hyperparameters
# Number of topics, currently same as the number of flairs
NUM_TOPICS = 14
 # This controls the how sparse the number of topics per document is
ALPHA = 0.001
# This controls how sparse the number of words per topic is
BETA = 0.1
VERSION = 7
NUM_ITERS = 1000
RANDOM_SEED = 42
BUILD_GIBBS_SAMPLING = False
LOAD_AND_ANALYZE = True
SPLIT = 0.9
SHUFFLE = True
TOP_WORDS_COUNT = 15
TOP_CUMULATIVE_PERCENTAGE = 0.15
ESCAPE_COUNT = 15

# Set random seed for reproducibility
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

# Load the documents this is assumed to be a dictionary with the a unique id
# For each document (block of text)
with open(DOC_PATH, 'r', encoding='utf-8') as f:
    documents = json.load(f)
for doc_id in documents:
    documents[doc_id] = documents[doc_id].split()

if SHUFFLE:
    doc_items = list(documents.items())
    random.shuffle(doc_items)
    documents = dict(doc_items)

# Load the word list
with open(WORD_LIST_PATH, 'r', encoding='utf-8') as f:
    global_word_counts = json.load(f)

word_list = list(global_word_counts['words'].keys()) \
          + list(global_word_counts['emojis'].keys())
# Remove stop words
word_list = [word for word in word_list if word not in STOP_WORDS]
word_map = {word: idx for idx, word in enumerate(word_list)}

# Load metadata from data
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    json_data = json.load(f)
flairs = {doc_id: json_data['meta'][doc_id]['flair'] for doc_id in documents}
unique_flairs = list(set(flairs.values()))

# Split the data into training and testing sets
num_train = int(SPLIT * len(documents))
train_docs =   {id: documents[id] for id in list(documents.keys())[:num_train]} if SPLIT else documents
test_docs =    {id: documents[id] for id in list(documents.keys())[num_train:]} if SPLIT else documents
train_flairs = {id: flairs[id] for id in list(documents.keys())[:num_train]}    if SPLIT else flairs
test_flairs =  {id: flairs[id] for id in list(documents.keys())[num_train:]}    if SPLIT else flairs

def initialize(_documents, _word_map, num_topics):
    """
    Randomly initialize topic assignments for each word in each document.
    
    Args:
        _documents (dict): A dictionary of documents where each document is a list of words.
        _word_map (dict): A mapping of words to their corresponding indices.
        num_topics (int): The number of topics.
        
    Returns:
        _topic_assignments (list): List of topic assignments for each document.
        _doc_topic_counts (ndarray): Document-topic counts matrix.
        _topic_word_counts (ndarray): Topic-word counts matrix.
        _topic_counts (ndarray): Total topic counts across all documents.
    """
    # Get the number of documents and vocabulary size
    num_docs = len(_documents)
    vocab_size = len(_word_map)

    # Initialize matrices to store counts:
    # - _doc_topic_counts stores counts of topics assigned to each document.
    # - _topic_word_counts stores counts of words assigned to each topic.
    # - _topic_counts stores the overall number of words assigned to each topic.
    _doc_topic_counts = np.zeros((num_docs, num_topics), dtype=int)
    _topic_word_counts = np.zeros((num_topics, vocab_size), dtype=int)
    _topic_counts = np.zeros(num_topics, dtype=int)

    # Used to keep track of the document index in the array
    doc_idx_list = list(_documents.keys())

    # Initialize a list to store topic assignments for each word in each document
    _topic_assignments = []

    # Loop through each document in the dataset
    for _doc_id, doc in _documents.items():
        # Get the document index from the list
        _doc_idx = doc_idx_list.index(_doc_id)
        print(f'Initializing document {_doc_idx + 1}/{num_docs}', end='\r')

        # Convert words in the document to their corresponding indices (skip words not in the vocabulary)
        word_indices = [_word_map[word] for word in doc if word in _word_map]

        # Assign a random topic for each word in the document
        topics = np.random.randint(0, num_topics, size=len(word_indices))

        # Store the topic assignments for this document
        _topic_assignments.append(topics.tolist())

        # Add 1 to the corresponding counts for each word-topic assignment
        np.add.at(_doc_topic_counts[_doc_idx], topics, 1)

        # Add 1 to the _topic_word_counts matrix at the positions of (topics, word_indices)
        np.add.at(_topic_word_counts, (topics, word_indices), 1)

        # Add 1 to the topic counts for each topic (word count)
        np.add.at(_topic_counts, topics, 1)

    print(f'Initialization of documents complete ({num_docs})')
    return _topic_assignments, _doc_topic_counts, _topic_word_counts, _topic_counts

def calc_log_likelihood(_doc_topic_counts, _topic_word_counts, alpha, beta):
    """
    Calculate the log likelihood of the current model state.

    Args:
        _doc_topic_counts (ndarray): Document-topic counts matrix.
        _topic_word_counts (ndarray): Topic-word counts matrix.
        alpha (float): Dirichlet prior for topic distribution in documents.
        beta (float): Dirichlet prior for word distribution in topics.

    Returns:
        log_likelihood (float): The log likelihood of the current model state.
    """
    log_likelihood = 0.0

    # Calculate the log likelihood for document-topic distribution
    for d in range(_doc_topic_counts.shape[0]):
        log_likelihood += (gammaln(np.sum(_doc_topic_counts[d]) + np.sum(alpha)) - \
            np.sum(gammaln(_doc_topic_counts[d] + alpha)))/np.sum(_doc_topic_counts[d])

    # Calculate the log likelihood for topic-word distribution
    for k in range(_topic_word_counts.shape[0]):
        log_likelihood += (gammaln(np.sum(_topic_word_counts[k]) + np.sum(beta)) - \
            np.sum(gammaln(_topic_word_counts[k] + beta)))/np.sum(_topic_word_counts[k])

    return log_likelihood

def gibbs_sampling(_documents, _word_map, num_topics, num_iters, alpha, beta):
    """
    Perform Gibbs sampling to train an LDA model.

    Args:
        _documents (dict): A dictionary of documents where each document is a list of words.
        _word_map (dict): A list of words in the corpus. (Non-used words are skipped)
        num_topics (int): The number of topics.
        num_iters (int): The number of iterations to perform.
        alpha (float): Dirichlet prior for the topic distribution in documents.
        beta (float): Dirichlet prior for the word distribution in topics.
        
    Returns:
        _doc_topic_counts (ndarray): Document-topic counts matrix.
        _topic_word_counts (ndarray): Topic-word counts matrix.
        _topic_counts (ndarray): Total topic counts across all documents.
        _topic_assignments (list): List of topic assignments for each document.
    """
    # Initialize counts and variables
    vocab_size = len(_word_map)
    # Get the document indices for reference
    _doc_idx = {doc_id: idx for idx, doc_id in enumerate(_documents.keys())}

    # Initialize random assignments
    init_values = initialize(documents, _word_map, num_topics)
    _topic_assignments, _doc_topic_counts, _topic_word_counts, _topic_counts = init_values

    # Initialize log likelihood for convergence check
    best_log_likelihood = np.inf

    # The words of the docs don't change so we can create a lookup table once for each doc
    word_idx_lookup = {}
    for _doc_id, doc in _documents.items():
        word_indices = [_word_map[word] for word in doc if word in _word_map]
        word_idx_lookup[_doc_id] = word_indices

    # Start Gibbs sampling for a set number of iterations
    for iter in range(num_iters):
        # Start time hack
        start_time = time.time()

        # Loop over each document and word to resample topics
        doc_count = 0
        for _doc_id, doc in _documents.items():
            print(f"Gibbs Sampling iter {iter + 1}/{num_iters} - Document {doc_count}", end='\r')
            doc_count += 1
            doc_index = _doc_idx[_doc_id]

            # Get word topic assignments for the document
            topics = _topic_assignments[doc_index]

            # Iterate over each word in the document
            for word_pos, word_idx in enumerate(word_idx_lookup[_doc_id]):
                current_topic = topics[word_pos]

                # Temporarily decrement counts to calculate the topic probabilities
                _doc_topic_counts[doc_index][current_topic] -= 1
                _topic_word_counts[current_topic][word_idx] -= 1
                _topic_counts[current_topic] -= 1

                # Recalculates the probabilities of each topic being assigned to this word
                topic_probs = (
                    (_doc_topic_counts[doc_index] + alpha)
                    * (_topic_word_counts[:, word_idx] + beta)
                    / (_topic_counts + beta * vocab_size)
                )
                # Normalize the probabilities
                topic_probs /= np.sum(topic_probs)

                # Resample a new topic based on the calculated probabilities
                new_topic = np.random.choice(np.arange(num_topics), p=topic_probs)

                # Update counts with the new topic assignment
                topics[word_pos] = new_topic
                _doc_topic_counts[doc_index][new_topic] += 1
                _topic_word_counts[new_topic][word_idx] += 1
                _topic_counts[new_topic] += 1

        # Calculate log-likelihood to check for convergence
        log_likelihood = calc_log_likelihood(_doc_topic_counts, _topic_word_counts, alpha, beta)

        # Check for convergence (exits after 5 consecutive iterations with no significant change)

        time_elapsed = time.time() - start_time
        common_message = (
            f"Gibbs Sampling Iteration {iter + 1}/{num_iters} - "
            f"Log Likelihood: {log_likelihood:.4f} - "
            f"Time: {time_elapsed:.2f}s"
        )

        if log_likelihood < best_log_likelihood:
            best_log_likelihood = log_likelihood
            escape_count = 0
            print(common_message)
        else:
            escape_count += 1
            print(f"{common_message} - Count: {escape_count}")
            if escape_count == ESCAPE_COUNT:
                print(f"\nConverged after {iter + 1} iterations.")
                break

    print("Gibbs sampling complete.")
    return _doc_topic_counts, _topic_word_counts, _topic_counts, _topic_assignments

def doc_topic_prediction(_documents, _word_map, _topic_word_counts, _topic_counts, beta):
    """
    Predict the most likely topic for each document using aggregated word probabilities.

    Args:
        _documents (dict): A dictionary of documents where each document is a list of words.
        _word_map (dict): A list of words in the corpus. (Non-used words are skipped)
        _topic_word_counts (ndarray): Topic-word counts matrix.
        _topic_counts (ndarray): Total topic counts across all documents.
        beta (float): Dirichlet prior for word distribution in topics.

    Returns:
        _predicted_topics (list): A list of predicted topics for each document.
    """
    vocab_size = _topic_word_counts.shape[1]
    num_topics = _topic_word_counts.shape[0]

    _predicted_topics = []

    for _, doc_words in _documents.items():
        # Initialize a probability array for topics in this document
        topic_probs = np.zeros(num_topics)

        # For each word in the document, calculate the topic probabilities
        for word in doc_words:
            if word in _word_map:  # Ensure the word is in our vocab
                word_idx = _word_map[word]

                # Calculate word probabilities across all topics
                word_topic_probs = (
                    (_topic_word_counts[:, word_idx] + beta)
                    / (_topic_counts + beta * vocab_size)
                )

                # Aggregate the word-topic probabilities
                topic_probs += word_topic_probs

        # Normalize the probabilities (this step isn't strictly necessary for argmax)
        topic_probs /= np.sum(topic_probs)

        # Predict the topic with the highest aggregated probability
        most_likely_topic = np.argmax(topic_probs)
        _predicted_topics.append(most_likely_topic)

    return _predicted_topics

def perplexity_est(_documents, _word_map, _topic_word_counts, _topic_counts, beta):
    """ 
    Calculate the perplexity of the model given a test set of documents.

    Args:
        _documents (dict): A dictionary of documents where each document is a list of words.
        _word_map (dict): A dict of words in the corpus. (Non-used words are skipped)
        _topic_word_counts (ndarray): Topic-word counts matrix.
        _topic_counts (ndarray): Total topic counts across all documents.
        _alpha (float): Dirichlet prior for topic distribution in documents.
        beta (float): Dirichlet prior for word distribution in topics.
    Returns:
        perplexity (float): The perplexity of the model.
    """
    vocab_size = _topic_word_counts.shape[1]
    num_topics = _topic_word_counts.shape[0]

    log_likelihood = 0.0
    word_count = 0
    for _, doc_words in _documents.items():
        # Initialize a probability array for topics in this document
        topic_probs = np.zeros(num_topics)

        # For each word in the document, calculate the topic probabilities
        for word in doc_words:
            # Ensure the word is in our vocab
            if word in _word_map:
                word_idx = _word_map[word]

                # Calculate word probabilities across all topics
                word_topic_probs = (
                    (_topic_word_counts[:, word_idx] + beta) 
                    / (_topic_counts + beta * vocab_size)
                )

                # Aggregate the word-topic probabilities
                topic_probs += word_topic_probs

                # Calculate the log likelihood for this word
                log_likelihood += np.log(np.sum(word_topic_probs))

                # Increment the word count
                word_count += 1

        # Normalize the probabilities (this step isn't strictly necessary for argmax)
        topic_probs /= np.sum(topic_probs)

    # Return the perplexity
    return np.exp(-log_likelihood / word_count)

def top_words_per_topic(_topic_word_counts, _word_list, num_words=10):
    """
    Get the top words for each topic based on word probabilities.

    Args:
        _topic_word_counts (ndarray): Topic-word counts matrix.
        _word_list (list): A list of words in the corpus.
        num_words (int): The number of top words to return for each topic.

    Returns:
        _top_words (list): A list of top words for each topic.
    """
    _top_words = []
    for topic_word_counts in _topic_word_counts:
        # Get the indices of the top words
        top_word_indices = np.argsort(topic_word_counts)[::-1][:num_words]
        # Get the actual words from the indices
        top_words = [_word_list[idx] for idx in top_word_indices]
        _top_words.append(top_words)
    return _top_words

def top_percent_per_topic(_topic_word_counts, _word_list, percentage=0.5):
    """
    Get the top words 'percentage' by cumulative probability for each topic.

    Args:
        _topic_word_counts (ndarray): Topic-word counts matrix.
        _word_list (list): A list of words in the corpus.
        percentage (float): The percentage of top words to return for each topic.

    Returns:
        _top_words (list): A list of top words for each topic.
    """
    _top_words = []
    for counts in _topic_word_counts:
        # Get the indices of the top words
        top_word_indices = np.argsort(counts)[::-1]
        # Get the actual words from the indices
        cumulative_prob = np.cumsum(counts[top_word_indices]) / np.sum(counts)
        top_word_indices = top_word_indices[cumulative_prob < percentage]
        _top_words.append([_word_list[idx] for idx in top_word_indices])
    return _top_words

def purity_est(_documents, _predicted_topics, _flairs, _unique_flairs):
    """
    Calculate the purity score for the LDA model.

    Args:
        _documents (dict): A dictionary where each key is a document ID and the value is the document text.
        _predicted_topics (list): A list of topic assignments for each document.
        _flairs (dict): A dictionary where each key is a document ID and the value is the flair (true class) for that document.
        _unique_flairs (list): A list of unique flairs (classes) in the dataset.

    Returns:
        float: The purity score of the topic model.
        topic_purity (ndarray): The purity score for each topic.
    """
    num_docs = len(_documents)
    num_topics = len(set(_predicted_topics))
    num_flairs = len(_unique_flairs)

    # Initialize a confusion matrix to store the number of documents per topic and flair
    confusion_matrix = np.zeros((num_topics, num_flairs))

    # Fill the confusion matrix
    for _doc_idx, _doc_id in enumerate(_documents):
        _topic = _predicted_topics[_doc_idx]
        _flair = _flairs[_doc_id]
        _flair_idx = _unique_flairs.index(_flair)
        confusion_matrix[_topic, _flair_idx] += 1

    # Calculate the purity score
    _purity = np.sum(np.max(confusion_matrix, axis=1)) / num_docs

    # Calculate the purity score for each topic
    _topic_purity = np.max(confusion_matrix, axis=1) / np.sum(confusion_matrix, axis=1)

    return _purity, _topic_purity

# Run the Gibbs sampling
if BUILD_GIBBS_SAMPLING:
    gibbs_output = gibbs_sampling(train_docs, word_map, NUM_TOPICS, NUM_ITERS, ALPHA, BETA)
    doc_topic_counts, topic_word_counts, topic_counts, topic_assignments = gibbs_output
    topic_assignments = [[int(topic) for topic in doc_topics] for doc_topics in topic_assignments]

    # Save the trained model
    model = {
        'doc_topic_counts': doc_topic_counts.tolist(),
        'topic_word_counts': topic_word_counts.tolist(),
        'topic_counts': topic_counts.tolist(),
        'topic_assignments': topic_assignments,
        'alpha': ALPHA,
        'beta': BETA
    }
    with open('lda_model' + str(VERSION) + '.json', 'w', encoding='utf-8') as f:
        json.dump(model, f)

# Load and analyze the trained model
if LOAD_AND_ANALYZE:
    with open('lda_model' + str(VERSION) + '.json', 'r', encoding='utf-8') as f:
        model = json.load(f)

    # Load the model components
    doc_topic_counts = np.array(model['doc_topic_counts'])
    topic_word_counts = np.array(model['topic_word_counts'])
    topic_counts = np.array(model['topic_counts'])
    topic_assignments = model['topic_assignments']

    # Try to load alpha and beta from the model / set to default if not found
    _ALPHA = model.get('alpha', ALPHA)
    _BETA = model.get('beta', BETA)
    print(f"Loaded model with ALPHA={_ALPHA} and BETA={_BETA}")

    # Predict the most likely topic for each document using aggregated word probabilities
    predicted_topics = doc_topic_prediction(documents, word_map, topic_word_counts, topic_counts, _BETA)

    # Initialize flair count per topic
    flair_counts_per_topic = np.zeros((NUM_TOPICS, len(unique_flairs)))

    # Fill the matrix with flair counts per topic
    for doc_idx, doc_id in enumerate(documents):
        topic = predicted_topics[doc_idx]
        flair = flairs[doc_id]
        flair_idx = unique_flairs.index(flair)
        flair_counts_per_topic[topic, flair_idx] += 1

    # Normalize the flair counts so that each topic's bar sums to 1
    topic_totals = flair_counts_per_topic.sum(axis=1, keepdims=True)  # Sum along flair axis
    flair_proportions_per_topic = flair_counts_per_topic / topic_totals  # Normalize

    # Create a stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.arange(NUM_TOPICS)

    # Plot the normalized flair proportions for each topic
    colors = [plt.cm.get_cmap('tab20b')(i / len(unique_flairs)) for i in range(len(unique_flairs))]
    bottom = np.zeros(NUM_TOPICS)
    for i, flair in enumerate(unique_flairs):
        ax.bar(indices, flair_proportions_per_topic[:, i], label=flair, bottom=bottom, color=colors[i])
        bottom += flair_proportions_per_topic[:, i]

    # Add labels and title
    ax.set_xlabel('Topic')
    ax.set_ylabel('Proportion of Documents')
    ax.set_title('Flair Distribution across Topics (Proportional)')
    ax.legend(title='Flair', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('flair_distribution_per_topic' + str(VERSION) + '.png', bbox_inches='tight')
    plt.close()

    # Stacked bar chart for flair_counts_per_topic
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.arange(NUM_TOPICS)

    # Plot each flair count for each topic in a stacked manner
    bottom = np.zeros(NUM_TOPICS)
    for i, flair in enumerate(unique_flairs):
        ax.bar(indices, flair_counts_per_topic[:, i], label=flair, bottom=bottom)
        bottom += flair_counts_per_topic[:, i]

    # Add labels and title
    ax.set_xlabel('Topic')
    ax.set_ylabel('Number of Documents')
    ax.set_title('Flair Distribution across Topics')
    ax.legend(title='Flair')
    plt.savefig('flair_per_topic' + str(VERSION) + '.png')


    # Perplexity estimation
    perplexity = perplexity_est(test_docs, word_map, topic_word_counts, topic_counts, _BETA)
    print(f"Perplexity: {perplexity:.4f}")

    # Purity estimation
    purity, topic_purity = purity_est(documents, predicted_topics, flairs, unique_flairs)
    print(f"Purity: {purity:.4f}")

    # Get the top words for each topic exported to a json file
    top_words = top_words_per_topic(topic_word_counts, word_list, num_words=TOP_WORDS_COUNT)
    top_percent = top_percent_per_topic(topic_word_counts, word_list, percentage=TOP_CUMULATIVE_PERCENTAGE)
    top_export = {}
    for topic_idx, words in enumerate(top_words):
        top_export[topic_idx] = {
            'to_N_word_list' : words,
            'purity' : topic_purity[topic_idx],
            'top_percent' : top_percent[topic_idx],
            'top_percent_count' : len(top_percent[topic_idx])
        }

    with open('top_words_per_topic' + str(VERSION) + '.json', 'w', encoding='utf-8') as f:
        json.dump(top_export, f)
