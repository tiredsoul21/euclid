""" Latent Dirichlet Allocation (LDA) model training using Gibbs Sampling. """
import json
import random
from collections import defaultdict

import numpy as np

DATA_PATH = '/home/derrick/data/reddit/teachers/Teachers.json'
WORD_LIST_PATH = '/home/derrick/data/reddit/teachers/wordlist.json'
DOC_PATH = '/home/derrick/data/reddit/teachers/documents.json'
NUM_TOPICS = 14 # Same as the number of flairs
ALPHA = 0.1  # Dirichlet prior for topic distribution in documents
BETA = 0.01  # Dirichlet prior for word distribution in topics
NUM_ITERS = 1000
RANDOM_SEED = 42


if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

# Load the documents
with open(DOC_PATH, 'r', encoding='utf-8') as f:
    documents = json.load(f)
for doc_id in documents:
    documents[doc_id] = documents[doc_id].split()

# Load the word list
with open(WORD_LIST_PATH, 'r', encoding='utf-8') as f:
    global_word_counts = json.load(f)

def initialize(_documents, num_topics, word_counts):
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    # Initialize topic assignments
    _doc_topic_counts = defaultdict(lambda: np.zeros(NUM_TOPICS))
    _topic_word_counts = defaultdict(lambda: np.zeros(len(global_word_counts['words'])
                                                + len(global_word_counts['emojis'])))
    _topic_counts = np.zeros(num_topics, dtype=np.int32)
    _topic_assignments = []
    word_list = list(word_counts['words'].keys())
    word_to_index = {word: idx for idx, word in enumerate(word_list)}

    count = 0
    for _doc_id, doc in _documents.items():
        count += 1
        if count == 50:
            break
        print(f'Initializing document {_doc_id} ({count}/{len(_documents)})')
        
        topics = np.random.randint(0, num_topics, size=len(doc))
        _doc_topic_counts[_doc_id] = np.bincount(topics, minlength=num_topics)
        _topic_counts += _doc_topic_counts[_doc_id]

        _topic_assignments.append(topics)

        for word_idx, word in enumerate(doc):
            if word in word_to_index:
                topic = topics[word_idx]
                word_index = word_to_index[word]
                _topic_word_counts[topic][word_index] += 1

    return _topic_assignments, _topic_counts, _doc_topic_counts, _topic_word_counts

def initialize2(documents, num_topics, word_counts):
    """ Initialize topic assignments randomly. """
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    _doc_topic_counts = defaultdict(lambda: np.zeros(num_topics))
    _topic_word_counts = defaultdict(lambda: np.zeros(len(word_counts['words'])))
    _topic_counts = np.zeros(num_topics, dtype=np.int32)

    _topic_assignments = []
    count = 0
    for doc in documents:
        count += 1
        if count == 50:
            break
        topics = []
        for word in doc:
            topic = random.randint(0, num_topics - 1)
            topics.append(topic)
            # Update counts for initialization
            _doc_topic_counts[doc][topic] += 1
            word_idx = list(word_counts['words'].keys()).index(word)
            _topic_word_counts[topic][word_idx] += 1
            _topic_counts[topic] += 1
        topic_assignments.append(topics)
    return _topic_assignments, _topic_counts, _doc_topic_counts, _topic_word_counts

def gibbs_sampling(_documents, num_topics, num_iterations, alpha, beta):
    """ Run Gibbs Sampling to infer topic assignments. """
    init_values = initialize(_documents, num_topics, global_word_counts)
    _topic_assignments, topic_counts, doc_topic_counts, topic_word_counts = init_values

    word_list = list(global_word_counts['words'].keys())  # Cache the word list

    for _ in range(num_iterations):
        print(f'Iteration {_ + 1}/{num_iterations}')
        for did, doc in enumerate(_documents):
            topics = _topic_assignments[did]
            for word_id, word in enumerate(doc):
                current_topic = topics[word_id]
                # Remove current word-topic assignment
                doc_topic_counts[doc][current_topic] -= 1

                # Skip words that are not in the word list
                if word in word_list:
                    word_idx = word_list.index(word)
                else:
                    # Log missing word or handle differently
                    continue  # Skip this word and move to the next one

                topic_word_counts[current_topic][word_idx] -= 1
                topic_counts[current_topic] -= 1

                # Compute topic probabilities
                topic_probs = np.zeros(num_topics)
                for topic in range(num_topics):
                    topic_prob = (doc_topic_counts[doc][topic] + alpha) \
                                 * (topic_word_counts[topic][word_idx] + beta) \
                                 / (topic_counts[topic] + beta * len(global_word_counts['words']))
                    topic_prob /= (doc_topic_counts[doc].sum() + alpha * num_topics)
                    topic_probs[topic] = topic_prob

                # Normalize and sample new topic
                topic_probs /= topic_probs.sum()

                # check for negative values

                new_topic = np.random.choice(range(num_topics), p=topic_probs)

                # Assign new topic
                topics[word_id] = new_topic
                doc_topic_counts[doc][new_topic] += 1
                topic_word_counts[new_topic][word_idx] += 1
                topic_counts[new_topic] += 1

    return _topic_assignments, doc_topic_counts, topic_word_counts, topic_counts


# Run Gibbs Sampling
topic_assignments, doc_topic_counts, topic_word_counts, topic_counts = gibbs_sampling(documents, NUM_TOPICS, NUM_ITERS, ALPHA, BETA)

def compute_topic_distributions(_doc_topic_counts, num_topics, alpha):
    """ Compute the topic distribution for each document. """
    return (_doc_topic_counts + alpha) / (_doc_topic_counts.sum(axis=0) + alpha * num_topics)

def compute_word_distributions(_topic_word_counts, _topic_counts, beta):
    """ Compute the word distribution for each topic. """
    vocab_size = len(global_word_counts['words']) + len(global_word_counts['emojis'])
    return (_topic_word_counts + beta) / (_topic_counts[:, None] + beta * vocab_size)

topic_distributions = compute_topic_distributions(doc_topic_counts, NUM_TOPICS, ALPHA)
word_distributions = compute_word_distributions(topic_word_counts, topic_counts, BETA)
