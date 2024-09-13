""" Latent Dirichlet Allocation (LDA) model training using Gibbs Sampling. """
import json
import random
from collections import defaultdict
from scipy.special import gammaln

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

def gibbs_sampling(_documents, word_list, num_topics, num_iters, alpha, beta, convergence_threshold=1e-3):
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

    # Initialize random assignments
    init_values = initialize(documents, word_to_index, num_topics)
    _topic_assignments, _doc_topic_counts, _topic_word_counts, _topic_counts = init_values

    # Initialize log likelihood for convergence check
    prev_log_likelihood = -np.inf

    # Start Gibbs sampling for a set number of iterations
    for iteration in range(num_iters):

        # Loop over each document and word to resample topics
        doc_count = 0
        for _doc_id, doc in _documents.items():
            print(f"Gibbs Sampling Iteration {iteration + 1}/{num_iters} - Document {doc_count}", end='\r')
            doc_count += 1
            doc_index = doc_idx.index(_doc_id)  # Get the document index
            word_indices = [word_to_index[word] for word in doc if word in word_to_index]

            # Get current topic assignments for the document
            topics = _topic_assignments[doc_index]

            # Vectorized decrements for all topics in the document
            np.add.at(_doc_topic_counts[doc_index], topics, -1)
            np.add.at(_topic_word_counts, (topics, word_indices), -1)
            np.add.at(_topic_counts, topics, -1)

            # Vectorized calculation of topic probabilities
            topic_probs = (
                (_doc_topic_counts[doc_index][:, None] + alpha)
                * (_topic_word_counts[:, word_indices] + beta) 
                / (_topic_counts[:, None] + beta * vocab_size)
            )
            topic_probs /= np.sum(topic_probs, axis=0)

            # Sample new topics for each word based on the calculated probabilities
            new_topics = np.array([np.random.choice(np.arange(num_topics), p=topic_probs[:, i])
                                   for i in range(len(word_indices))])

            # Update counts with the new topic assignments
            np.add.at(_doc_topic_counts[doc_index], new_topics, 1)
            np.add.at(_topic_word_counts, (new_topics, word_indices), 1)
            np.add.at(_topic_counts, new_topics, 1)

            # Store the new topics in the topic assignments
            _topic_assignments[doc_index] = new_topics

            # Iterate over each word in the document
            # for word_pos, word_idx in enumerate(word_indices):
            #     current_topic = topics[word_pos]

            #     # Decrement counts for the current topic assignment
            #     _doc_topic_counts[doc_index][current_topic] -= 1
            #     _topic_word_counts[current_topic][word_idx] -= 1
            #     _topic_counts[current_topic] -= 1

            #     # Step 4: Calculate the topic probabilities using the current state
            #     topic_probs = (
            #         (_doc_topic_counts[doc_index] + alpha)  # Document-topic prior
            #         * (_topic_word_counts[:, word_idx] + beta)  # Topic-word prior
            #         / (_topic_counts + beta * vocab_size)  # Normalization for topic-word counts
            #     )

            #     # Normalize the probabilities
            #     topic_probs /= np.sum(topic_probs)

            #     # Step 5: Resample a new topic based on the calculated probabilities
            #     new_topic = np.random.choice(np.arange(num_topics), p=topic_probs)

            #     # Step 6: Update counts with the new topic assignment
            #     topics[word_pos] = new_topic
            #     _doc_topic_counts[doc_index][new_topic] += 1
            #     _topic_word_counts[new_topic][word_idx] += 1
            #     _topic_counts[new_topic] += 1

            # print(_topic_assignments[doc_index])
            # exit()
        # Calculate log-likelihood to check for convergence
        log_likelihood = calc_log_likelihood(_doc_topic_counts, _topic_word_counts, _topic_counts, alpha, beta, vocab_size)

        # Check for convergence
        if abs(log_likelihood - prev_log_likelihood) < convergence_threshold:
            print(f"Converged after {iteration + 1} iterations.")
            break
        else:
            print(f"Gibbs Sampling Iteration {iteration + 1}/{num_iters} - Log Likelihood: {log_likelihood:.4f}")
        
        prev_log_likelihood = log_likelihood

    print("Gibbs sampling complete.")
    return _doc_topic_counts, _topic_word_counts, _topic_counts, _topic_assignments

# Run the Gibbs sampling
gibbs_output = gibbs_sampling(documents, global_word_counts, NUM_TOPICS, NUM_ITERS, ALPHA, BETA)


# Save the trained model
doc_topic_counts, topic_word_counts, topic_counts, topic_assignments = gibbs_output
model = {
    'doc_topic_counts': doc_topic_counts.tolist(),
    'topic_word_counts': topic_word_counts.tolist(),
    'topic_counts': topic_counts.tolist(),
    'topic_assignments': topic_assignments
}
with open('lda_model1.json', 'w', encoding='utf-8') as f:
    json.dump(model, f)
