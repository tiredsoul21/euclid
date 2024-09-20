""" Latent Dirichlet Allocation (LDA) model training using Gibbs Sampling. """
import json
import random
import time

from scipy.special import gammaln
import matplotlib.pyplot as plt

import numpy as np

# Reddit data (Not used)
DATA_PATH = '/home/derrick/data/reddit/teachers/Teachers.json'
# The list of words as keys.  Should have the global word usage pery word
WORD_LIST_PATH = '/home/derrick/data/reddit/teachers/wordlist.json'
# The key(id) value(text) pair of the documents
DOC_PATH = '/home/derrick/data/reddit/teachers/documents.json'

# LDA hyperparameters
NUM_TOPICS = 14 # Same as the number of flairs
ALPHA = 0.1  # Dirichlet prior for topic distribution in documents
BETA = 0.1  # Dirichlet prior for word distribution in topics
NUM_ITERS = 1000
RANDOM_SEED = 42
BUILD_GIBBS_SAMPLING = False
LOAD_AND_ANALYZE = True

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

# Load the word list
with open(WORD_LIST_PATH, 'r', encoding='utf-8') as f:
    global_word_counts = json.load(f)

# Load metadata from data
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    json_data = json.load(f)
flairs = {doc_id: json_data['meta'][doc_id]['flair'] for doc_id in documents}
unique_flairs = list(set(flairs.values()))

def initialize(_documents, word_to_index, num_topics):
    """
    Randomly initialize topic assignments for each word in each document.
    
    Args:
        _documents (dict): A dictionary of documents where each document is a list of words.
        word_to_index (dict): A mapping of words to their corresponding indices.
        num_topics (int): The number of topics.
        
    Returns:
        _topic_assignments (list): List of topic assignments for each document.
        _doc_topic_counts (ndarray): Document-topic counts matrix.
        _topic_word_counts (ndarray): Topic-word counts matrix.
        _topic_counts (ndarray): Total topic counts across all documents.
    """

    # Get the number of documents
    num_docs = len(_documents)

    # Calculate the size of the vocabulary (words + emojis)
    vocab_size = len(global_word_counts['words']) + len(global_word_counts['emojis'])

    # Initialize matrices to store counts:
    # - _doc_topic_counts stores counts of topics assigned to each document.
    # - _topic_word_counts stores counts of words assigned to each topic.
    # - _topic_counts stores the overall number of words assigned to each topic.
    _doc_topic_counts = np.zeros((num_docs, num_topics), dtype=int)
    _topic_word_counts = np.zeros((num_topics, vocab_size), dtype=int)
    _topic_counts = np.zeros(num_topics, dtype=int)

    # Used to keep track of the document index in the array
    doc_idx = list(_documents.keys())

    # Initialize a list to store topic assignments for each word in each document
    _topic_assignments = []

    # Loop through each document in the dataset
    for _doc_id, doc in _documents.items():
        print(f'Initializing document {doc_idx.index(_doc_id) + 1}/{num_docs}', end='\r')

        # Convert words in the document to their corresponding indices (skip words not in the vocabulary)
        word_indices = [word_to_index[word] for word in doc if word in word_to_index]

        # Assign a random topic for each word in the document
        topics = np.random.randint(0, num_topics, size=len(word_indices))

        # Store the topic assignments for this document
        _topic_assignments.append(topics.tolist())

        np.add.at(_doc_topic_counts[doc_idx.index(_doc_id)], topics, 1)
        np.add.at(_topic_word_counts, (topics, word_indices), 1)
        np.add.at(_topic_counts, topics, 1)

    print(f'Initialization of documents complete ({num_docs})')
    return _topic_assignments, _doc_topic_counts, _topic_word_counts, _topic_counts

def calc_log_likelihood(doc_topic_counts, topic_word_counts, topic_counts, alpha, beta, vocab_size):
    """
    Calculate the log likelihood of the current model state.

    Args:
        doc_topic_counts (ndarray): Document-topic counts matrix.
        topic_word_counts (ndarray): Topic-word counts matrix.
        topic_counts (ndarray): Total topic counts across all documents.
        alpha (float): Dirichlet prior for topic distribution in documents.
        beta (float): Dirichlet prior for word distribution in topics.
        vocab_size (int): Size of the vocabulary.

    Returns:
        log_likelihood (float): The log likelihood of the current model state.
    """
    log_likelihood = 0.0

    # Calculate the log likelihood for document-topic distribution
    for d in range(doc_topic_counts.shape[0]):
        log_likelihood += (gammaln(np.sum(doc_topic_counts[d]) + np.sum(alpha)) - \
                          np.sum(gammaln(doc_topic_counts[d] + alpha)))/np.sum(doc_topic_counts[d])

    # Calculate the log likelihood for topic-word distribution
    for k in range(topic_word_counts.shape[0]):
        log_likelihood += (gammaln(np.sum(topic_word_counts[k]) + np.sum(beta)) - \
                          np.sum(gammaln(topic_word_counts[k] + beta)))/np.sum(topic_word_counts[k])

    return log_likelihood

def gibbs_sampling(_documents, word_list, num_topics, num_iters, alpha, beta):
    """
    Perform Gibbs sampling to train an LDA model.

    Args:
        _documents (dict): A dictionary of documents where each document is a list of words.
        word_list (list): A list of words in the corpus. (Non-used words are skipped)
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
    vocab_size = len(word_list['words']) + len(word_list['emojis'])
    _word_list = list(word_list['words'].keys()) + list(word_list['emojis'].keys())
    word_to_index = {word: idx for idx, word in enumerate(_word_list)}  # Word to index map
    doc_idx = list(_documents.keys())  # Get the document indices for reference
    print(f"Vocabulary size: {vocab_size}")
    # Initialize random assignments
    init_values = initialize(documents, word_to_index, num_topics)
    _topic_assignments, _doc_topic_counts, _topic_word_counts, _topic_counts = init_values

    # Initialize log likelihood for convergence check
    best_log_likelihood = np.inf

    # Create word index mapping for faster access
    word_idx_lookup = {}
    for _doc_id, doc in _documents.items():
        word_indices = [word_to_index[word] for word in doc if word in word_to_index]
        word_idx_lookup[_doc_id] = word_indices

    # Start Gibbs sampling for a set number of iterations
    for iteration in range(num_iters):
        # Start time hack
        start_time = time.time()

        # Loop over each document and word to resample topics
        doc_count = 0
        for _doc_id, doc in _documents.items():
            print(f"Gibbs Sampling Iteration {iteration + 1}/{num_iters} - Document {doc_count}", end='\r')
            doc_count += 1
            doc_index = doc_idx.index(_doc_id)  # Get the document index

            # Get current topic assignments for the document
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
                    (_doc_topic_counts[doc_index] + alpha)  # Document-topic prior
                    * (_topic_word_counts[:, word_idx] + beta)  # Topic-word prior
                    / (_topic_counts + beta * vocab_size)  # Normalization for topic-word counts
                )
                topic_probs /= np.sum(topic_probs) # Normalize

                # Resample a new topic based on the calculated probabilities
                new_topic = np.random.choice(np.arange(num_topics), p=topic_probs)

                # Update counts with the new topic assignment
                topics[word_pos] = new_topic
                _doc_topic_counts[doc_index][new_topic] += 1
                _topic_word_counts[new_topic][word_idx] += 1
                _topic_counts[new_topic] += 1

        # Calculate log-likelihood to check for convergence
        log_likelihood = calc_log_likelihood(_doc_topic_counts, _topic_word_counts, _topic_counts, alpha, beta, vocab_size)

        # Check for convergence (exits after 5 consecutive iterations with no significant change)

        time_elapsed = time.time() - start_time
        if log_likelihood < best_log_likelihood:
            best_log_likelihood = log_likelihood
            escape_count = 0
            print(f"Gibbs Sampling Iteration {iteration + 1}/{num_iters} - Log Likelihood: {log_likelihood:.4f} - Time: {time_elapsed:.2f}s")
        else:
            escape_count += 1
            print(f"Gibbs Sampling Iteration {iteration + 1}/{num_iters} - Log Likelihood: {log_likelihood:.4f} - Time: {time_elapsed:.2f}s - Count: {escape_count}")
            if escape_count == 5:
                print(f"\nConverged after {iteration + 1} iterations.")
                break


        best_log_likelihood = log_likelihood

    print("Gibbs sampling complete.")
    return _doc_topic_counts, _topic_word_counts, _topic_counts, _topic_assignments

# Run the Gibbs sampling
if BUILD_GIBBS_SAMPLING:
    gibbs_output = gibbs_sampling(documents, global_word_counts, NUM_TOPICS, NUM_ITERS, ALPHA, BETA)
    doc_topic_counts, topic_word_counts, topic_counts, topic_assignments = gibbs_output
    topic_assignments = [[int(topic) for topic in doc_topics] for doc_topics in topic_assignments]

    # Save the trained model
    model = {
        'doc_topic_counts': doc_topic_counts.tolist(),
        'topic_word_counts': topic_word_counts.tolist(),
        'topic_counts': topic_counts.tolist(),
        'topic_assignments': topic_assignments
    }
    with open('lda_model1.json', 'w', encoding='utf-8') as f:
        json.dump(model, f)


    flair_counts_per_topic = np.zeros((NUM_TOPICS, len(unique_flairs)))


    # Fill the matrix with flair counts per topic
    for doc_idx, doc_id in enumerate(documents):
        topic = topic_assignments[doc_idx]
        flair = flairs[doc_id]
        flair_idx = unique_flairs.index(flair)
        flair_counts_per_topic[topic, flair_idx] += 1

    # Create a stacked bar chart
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

    plt.show()

def predict_document_topic_by_probs(_documents, word_list, _topic_word_counts, _topic_counts, _alpha, beta):
    vocab_size = _topic_word_counts.shape[1]
    num_topics = _topic_word_counts.shape[0]

    # Convert word_list to a word -> index dictionary
    _word_list = list(word_list['words'].keys()) + list(word_list['emojis'].keys())
    word_to_index = {word: idx for idx, word in enumerate(_word_list)}

    _predicted_topics = []
    
    for doc_id, doc_words in _documents.items():
        # Initialize a probability array for topics in this document
        topic_probs = np.zeros(num_topics)
        
        # For each word in the document, calculate the topic probabilities
        for word in doc_words:
            if word in word_to_index:  # Ensure the word is in our vocab
                word_idx = word_to_index[word]

                # Calculate word probabilities across all topics
                word_topic_probs = (
                    (_topic_word_counts[:, word_idx] + beta)  # Topic-word prior
                    / (_topic_counts + beta * vocab_size)  # Normalization for topic-word counts
                )

                # Aggregate the word-topic probabilities
                topic_probs += word_topic_probs

        # Normalize the probabilities (this step isn't strictly necessary for argmax)
        topic_probs /= np.sum(topic_probs)

        # Predict the topic with the highest aggregated probability
        most_likely_topic = np.argmax(topic_probs)
        _predicted_topics.append(most_likely_topic)

    return _predicted_topics

def perplexity_est(_documents, word_list, _topic_word_counts, _topic_counts, _alpha, beta):
    """ 
    Calculate the perplexity of the model given a test set of documents.

    Args:
        _documents (dict): A dictionary of documents where each document is a list of words.
        word_list (list): A list of words in the corpus. (Non-used words are skipped)
        _topic_word_counts (ndarray): Topic-word counts matrix.
        _topic_counts (ndarray): Total topic counts across all documents.
        _alpha (float): Dirichlet prior for topic distribution in documents.
        beta (float): Dirichlet prior for word distribution in topics.
    Returns:
        perplexity (float): The perplexity of the model.
    """
    vocab_size = _topic_word_counts.shape[1]
    num_topics = _topic_word_counts.shape[0]

    # Convert word_list to a word -> index dictionary
    _word_list = list(word_list['words'].keys()) + list(word_list['emojis'].keys())
    word_to_index = {word: idx for idx, word in enumerate(_word_list)}

    log_likelihood = 0.0
    word_count = 0

    for _, doc_words in _documents.items():
        # Initialize a probability array for topics in this document
        topic_probs = np.zeros(num_topics)

        # For each word in the document, calculate the topic probabilities
        for word in doc_words:
            if word in word_to_index:  # Ensure the word is in our vocab
                word_idx = word_to_index[word]

                # Calculate word probabilities across all topics
                word_topic_probs = (
                    (_topic_word_counts[:, word_idx] + beta)  # Topic-word prior
                    / (_topic_counts + beta * vocab_size)  # Normalization for topic-word counts
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

def purity_est(
    """
    Calculate the purity score for the LDA model.

    Args:
        _documents (dict): A dictionary where each key is a document ID and the value is the document text.
        _predicted_topics (list): A list of topic assignments for each document.
        _flairs (dict): A dictionary where each key is a document ID and the value is the flair (true class) for that document.
        _unique_flairs (list): A list of unique flairs (classes) in the dataset.

    Returns:
        float: The purity score of the topic model.
    """
    

# Load and analyze the trained model
if LOAD_AND_ANALYZE:
    with open('lda_model1.json', 'r', encoding='utf-8') as f:
        model = json.load(f)

    # Load the model components
    doc_topic_counts = np.array(model['doc_topic_counts'])
    topic_word_counts = np.array(model['topic_word_counts'])
    topic_counts = np.array(model['topic_counts'])
    topic_assignments = model['topic_assignments']

    # Predict the most likely topic for each document using aggregated word probabilities
    predicted_topics = predict_document_topic_by_probs(documents, global_word_counts, topic_word_counts, topic_counts, ALPHA, BETA)

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

    # Dynamically generate a list of unique colors based on the number of flairs
    colors = [plt.cm.get_cmap('tab20b')(i / len(unique_flairs)) for i in range(len(unique_flairs))]

    # Plot the normalized flair proportions for each topic
    bottom = np.zeros(NUM_TOPICS)
    for i, flair in enumerate(unique_flairs):
        ax.bar(indices, flair_proportions_per_topic[:, i], label=flair, bottom=bottom, color=colors[i])
        bottom += flair_proportions_per_topic[:, i]

    # Add labels and title
    ax.set_xlabel('Topic')
    ax.set_ylabel('Proportion of Documents')
    ax.set_title('Flair Distribution across Topics (Proportional)')

    # Place the legend outside the plot
    ax.legend(title='Flair', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()  # Adjust the plot to make space for the legend

    # Save the plot
    plt.savefig('flair_distribution_per_topic.png', bbox_inches='tight')

    # Perplexity estimation
    perplexity = perplexity_est(documents, global_word_counts, topic_word_counts, topic_counts, ALPHA, BETA)
    print(f"Perplexity: {perplexity:.4f}")