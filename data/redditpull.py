"""A script to pull posts and comments from a subreddit and store them in a JSON file."""
import json
import time

import praw

# Reddit API credentials
reddit = praw.Reddit(
    client_id="fARBajzjQn0YWiR51c9E9w",
    client_secret="xu4t8hP78Fi_Uawiwp-0yqu0_Oz9Wg",
    user_agent="PythonAPI",
)

# The number of posts to pull before stopping (so we can save the data)
PULL_LIMIT = 3000

# Output file location
OUTPUT_FILE = "/home/derrick/data/reddit/teachers/Teachers2.json"

# Subreddit to pull from
SUBREDDIT_NAME = "Teachers"

# Keyword to search for in the post title leave empty to skip
KEYWORD = ""

# List of flairs to pull from
flair_text_list=[
    "SUCCESS!",
    "Humor",
    "Teacher Support &/or Advice",
    "Student Teacher Support &/or Advice",
    "Career & Interview Advice",
    "Higher Ed / PD / Cert Exams",
    "Professional Dress & Wardrobe",
    "Classroom Management & Strategies",
    "Pedagogy & Best Practices",
    "Curriculum",
    "Policy & Politics",
    "Bad Teacher, No Apple",
    "Blatant Abuse of Power",
    "Moderator Announcement",
    "Student or Parent",
    "Power of Positivity",
    "New Teacher",
    "Just Smile and Nod Y'all.  ",
    "Non-US Teacher",
    "Rant & Vent",
    "Retired Teacher",
    "Another AI / ChatGPT Post 🤖",
    "Substitute Teacher",
    "Charter or Private School"
]

# Load the existing post IDs from json
flair_data = {}
meta_data = {}
pulled_ids = set()
total_size = 0
try:
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        flair_data = data['flair']
        meta_data = data['meta']
        for post_id in meta_data.keys():
            pulled_ids.add(post_id)
            total_size += 1
        print(f"Loaded {len(pulled_ids)} posts from the file")
except FileNotFoundError:
    print("No existing data file found, starting fresh.")
    pass

subreddit = reddit.subreddit(SUBREDDIT_NAME)

# Append new posts to the existing data
count_duplicates = 0
count_posts = 0
try:
    for flair_text in flair_text_list:
        print("---"*10)
        print(flair_text)
        print("---"*10)

        # Limit the number of posts pulled to make sure we save the data in a reasonable time
        if count_posts >= PULL_LIMIT:
            print("Reached pull limit")
            break

        # Create a search pattern to pull posts with the specified flair and KEYWORD
        search_pattern = f'flair:"{flair_text}" {KEYWORD}' if KEYWORD else f'flair:"{flair_text}"'

        # for post in subreddit.search(search_pattern, sort='hot', limit=None):
        # for post in subreddit.search(search_pattern, sort='top', limit=None): # This may break as comments are many!
        # for post in subreddit.search(search_pattern, sort='rising', limit=None):
        for post in subreddit.search(search_pattern, sort='new', limit=None):
            if count_posts >= PULL_LIMIT:
                break

            # Skip posts without comments
            if post.num_comments == 0:
                print(f"Skipping post without comments: {post.id}")
                continue

            if post.id not in pulled_ids:
                print(f"{count_posts} - {post.title}")
                flair = post.link_flair_text or "No Flair"  # Handle posts without flair
                post_id = post.id
                post_info = {
                    'title': post.title,
                    'author': str(post.author),
                    'score': post.score,
                    'text_body': post.selftext if post.is_self else post.url,
                    'num_comments': post.num_comments,
                    'comments': {}
                }

                # Fetch and store comments
                post.comments.replace_more(limit=None)
                for comment in post.comments.list():
                    if comment.body == '[deleted]':
                        continue
                    post_info['comments'][comment.id] = {
                        'comment': comment.body,
                        'score': comment.score,
                        'parentId': comment.parent_id.split('_')[1]
                    }

                # Add post info to the flair_data structure
                if flair not in flair_data:
                    flair_data[flair] = {}
                flair_data[flair][post_id] = post_info

                # Add post ID and flair to the meta_data structure
                meta_data[post_id] = {
                    'flair': flair,
                    'keywords': []
                }

                # Add keywords to the meta_data structure if KEYWORD is set
                if KEYWORD:
                    meta_data[post_id]['keywords'].append(KEYWORD)

                time.sleep(.5)
                count_posts += 1
                total_size += 1
            else:
                # If the post is already in the data, update keywords if necessary
                key_word_updated = False
                if KEYWORD and KEYWORD not in meta_data[post.id]['keywords']:
                    meta_data[post.id]['keywords'].append(KEYWORD)
                    key_word_updated = True

                count_duplicates += 1
                print(f"Duplicate post: {post.id} - Keywords updated: {key_word_updated}")
except Exception as e:
    print("An error happened, writing to file and then exiting")
    print(e)

output_data = {
    'meta': meta_data,
    'flair': flair_data
}

# Store the data in a JSON file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Posts pulled: {count_posts}")
print(f"Total size: {total_size}")
print(f"Duplicates skipped: {count_duplicates}")
