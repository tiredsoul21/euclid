""" This is to concatenate the data from multiple pulls into a single file """

import json

OUTPUT_FILE = "/home/derrick/data/reddit/teachers/Teachers_cat.json"
# Assumes the files are in order and will overwrite previous entries
INPUT_LIST = ["/home/derrick/data/reddit/teachers/Teachers.json", "/home/derrick/data/reddit/teachers/Teachers1.json"]


# Load the existing post IDs from json
flair_data = {}
meta_data = {}
pulled_ids = set()

for input_file in INPUT_LIST:
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            f_flair_data = data['flair']
            f_meta_data = data['meta']
            for post_id in f_meta_data.keys():
                pulled_ids.add(post_id)
                #combine unique keywords
                if post_id in meta_data:
                    meta_data[post_id]['keywords'] = list(set(meta_data[post_id]['keywords'] + f_meta_data[post_id]['keywords']))
                else:
                    meta_data[post_id] = f_meta_data[post_id]
                if f_meta_data[post_id]['flair'] not in flair_data:
                    flair_data[f_meta_data[post_id]['flair']] = {}
                flair_data[f_meta_data[post_id]['flair']][post_id] = f_flair_data[f_meta_data[post_id]['flair']][post_id]
            print(f"Loaded {len(pulled_ids)} posts from the file {input_file}")
            print(f"Total posts: {len(meta_data)}")
    except FileNotFoundError:
        pass

# Save the data to a json file
data = {'flair': flair_data, 'meta': meta_data}
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)
