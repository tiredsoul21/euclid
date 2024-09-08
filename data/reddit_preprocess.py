""" Latent Dirichlet Allocation (LDA) model training using Gibbs Sampling. """
import re
import sys
import json
import emoji

DATA_PATH = '/home/derrick/data/reddit/teachers/Teachers.json'
WORD_LIST_PATH = '/home/derrick/data/reddit/teachers/wordlist.json'
DOC_PATH = '/home/derrick/data/reddit/teachers/documents.json'
word_list = { 'words': {} , 'emojis': {}}
documents = {}
keep_words = ["#N", "#D"]

# Pruning options
REPLACE_EMOJIS = True
PRUNE_DOCUMENT_EXCLUSIVE_WORDS = True
PRUNE_RARE_WORDS = True
WORD_COUNT_THRESHOLD = 5
PRUNE_GLOBAL_USE_WORDS = True
GLOBAL_USE_THRESHOLD = 0.50

# Load the data
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    json_data = json.load(f)
if 'meta' not in json_data or 'flair' not in json_data:
    print("Invalid data format")
    sys.exit(1)
meta_data = json_data['meta']
json_data = json_data['flair']
post_ids = meta_data.keys()
flair_keys = json_data.keys()
print(f"Loaded {len(post_ids)} posts")
print(f"Flair keys: {flair_keys}")

# Document creation
for post_id in post_ids:
    post_meta = meta_data[post_id]
    flair = post_meta['flair']

    doc_json = json_data[flair][post_id]
    documents[post_id] = doc_json['title'] + ' ' + doc_json['text_body']
    for key in doc_json['comments'].keys():
        documents[post_id] += ' ' + doc_json['comments'][key]['comment']
print(f"Created {len(documents)} documents")

# Cast to lowercase
for document in documents:
    documents[document] = documents[document].lower()

# Replace emojis
if REPLACE_EMOJIS:
    for document in documents:
        documents[document] = emoji.demojize(documents[document])
    print("Replaced emojis")

# Remove special characters
known_chars = 'abcdefghijklmnopqrstuvwxyz0123456789 '
outlying_chars = set()
for document in documents:
    # Replace unicode characters
    documents[document] = documents[document].encode('ascii', 'ignore').decode('utf-8')

    # Remove URLs
    documents[document] = re.sub(r'\[.*?https?://\S+.*?\]', '', documents[document])
    documents[document] = re.sub(r'\S*https?://\S*\S*', '', documents[document])

    # Special characters
    special_list = ['\"', '\\', '/', '_', "'", '*', '(', ')', '^', '.', ',', '!', '?', '#', '@', '[', ']', '{', '}']
    for special in special_list:
        documents[document] = documents[document].replace(special, '')

    #
    documents[document] = documents[document].replace('$$$$', '$$$')
    documents[document] = documents[document].replace('$$$$$', '$$$')


    # Replace $Number with #D
    documents[document] = re.sub(r'\$\d+', '#D', documents[document])

    # Replace 100% numbers with #N
    documents[document] = re.sub(r'(?<!\S)\d+(?!\S)', '#N', documents[document])

    # Replace time with #T
    documents[document] = re.sub(r'\b(\d{1,2}):(\d{2})\b', '#T', documents[document])

    #print outlying characters
    for char in documents[document]:
        if char not in known_chars:
            outlying_chars.add(char)
# print("Removed special characters")
print(f"Outlying characters: {outlying_chars}")

# Create a dictionary of words and emojis
for doc in documents.values():
    words = doc.split()
    for word in words:
        # Check if the word is an emoji (starts & ends with :)
        if word[0] == ':' and word[-1] == ':':
            word_list['emojis'][word] = 1 if word not in word_list['emojis'] else word_list['emojis'][word] + 1
        else:
            word_list['words'][word] = 1 if word not in word_list['words'] else word_list['words'][word] + 1
            
print(f"Unique words: {len(word_list['words'])} Unique emojis: {len(word_list['emojis'])}")

# Prune words that are exclusive to a document
if PRUNE_DOCUMENT_EXCLUSIVE_WORDS:
    # Here we check if document word count == total word count
    for doc in documents:
        doc_words = { 'words': {} , 'emojis': {}}
        words = documents[doc].split()
        for word in words:
            if word[0] == ':' and word[-1] == ':':
                doc_words['emojis'][word] = 1 if word not in doc_words['emojis'] else doc_words['emojis'][word] + 1
            else:
                doc_words['words'][word] = 1 if word not in doc_words['words'] else doc_words['words'][word] + 1
        # Check if the document has exclusive words
        for word in doc_words['words']:
            if word_list['words'][word] == doc_words['words'][word]:
                del word_list['words'][word]
        for word in doc_words['emojis']:
            if word_list['emojis'][word] == doc_words['emojis'][word]:
                del word_list['emojis'][word]
    print("Pruned exclusive words")
    print(f"Unique words: {len(word_list['words'])} Unique emojis: {len(word_list['emojis'])}")

# Prune words with count less than threshold
if PRUNE_RARE_WORDS:
    for word in list(word_list['words'].keys()):
        if word_list['words'][word] < WORD_COUNT_THRESHOLD:
            del word_list['words'][word]
    for word in list(word_list['emojis'].keys()):
        if word_list['emojis'][word] < WORD_COUNT_THRESHOLD:
            del word_list['emojis'][word]
    print("Pruned rare words")
    print(f"Unique words: {len(word_list['words'])} Unique emojis: {len(word_list['emojis'])}")

# Prune words that are used in more than a certain percentage of documents
if PRUNE_GLOBAL_USE_WORDS:
    pruned_words = []
    doc_word_count = { 'words': {} , 'emojis': {}}
    for doc in documents.values():
        word_set = set(doc.split())
        for word in word_set:
            if word[0] == ':' and word[-1] == ':':
                doc_word_count['emojis'][word] = 1 if word not in doc_word_count['emojis'] else doc_word_count['emojis'][word] + 1
            else:
                doc_word_count['words'][word] = 1 if word not in doc_word_count['words'] else doc_word_count['words'][word] + 1
    for word in list(word_list['words'].keys()):
        # Skip words that are intentionally kept
        if word in keep_words:
            continue

        usage =  doc_word_count['words'][word] / len(documents)
        if usage > GLOBAL_USE_THRESHOLD:
            pruned_words.append(word)
            del word_list['words'][word]

    print("Pruned global use words")
    print(f"Pruned: {pruned_words}")
    print(f"Unique words: {len(word_list['words'])} Unique emojis: {len(word_list['emojis'])}")

# Export the dictionary to a file sorted by word count
word_list['words'] = dict(sorted(word_list['words'].items(), key=lambda item: item[1], reverse=True))
word_list['emojis'] = dict(sorted(word_list['emojis'].items(), key=lambda item: item[1], reverse=True))
with open(WORD_LIST_PATH, 'w', encoding='utf-8') as f:
    json.dump(word_list, f)

# Export the documents to a file
with open(DOC_PATH, 'w', encoding='utf-8') as f:
    json.dump(documents, f)