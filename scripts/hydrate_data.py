"""
Hydrates ../data/corpus.json by scraping the AT API for more dynamic data such as like counts, and creates a new JSON at ../data/hydrated_corpus.json.
Run this script before using data to ensure that everything is up to date.
"""
import json, requests, time
from pathlib import Path
from bluesky_scraper import print_progress

URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts"

DATA_DIR = Path("../data")
DATA_FILE = "corpus.json"
OUT_FILE = "hydrated_corpus.json"

DATA_PATH = DATA_DIR / DATA_FILE
OUT_PATH = DATA_DIR / OUT_FILE

BATCH_SIZE = 25

def get_batch(session, uris, retries = 4, timeout_s = 30, base_backoff = 0.8):
    """ 
    Requests the public Bluesky API for a batch of posts

    Parameters:
        session      - Required : Requests session (requests.Session)
        uris         - Required : List of URIs to query (list of strs)
        retries      - Optional : Number of times to retry before giving up (int)
        timeout_s    - Optional : Timeout on GET request in seconds (int)
        base_backoff - Optional : Base duration of time between retries (float)
    """
    params = {"uris" : [u for u in uris]}
    for attempt in range(retries + 1):
        try:
            resp = session.get(URL, params=params, timeout=timeout_s)

            # Successful
            if resp.status_code == 200:
                data = resp.json()
                return data.get("posts")
            
            # Error codes worth retrying
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    sleep_s = float(retry_after)
                else:
                    sleep_s = base_backoff * (2 ** attempt)

                if attempt < retries:
                    time.sleep(min(sleep_s, 30.0))
                    continue
                return []
            
            # Error codes not worth retrying
            return []
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(min(base_backoff * (2 ** attempt), 30.0))
                continue
            return []
    return []

def hydrate():
    """ 
    Hydrates the data found in ../data/corpus.json
    """
    with requests.Session() as session:
        with DATA_PATH.open("r") as f:
            dry_corpus = json.load(f)
        data = dry_corpus.get("data")
        corpus = []
        batch = []

        hydrated = 0
        i = 0
        print_progress(i, len(data) - 1, prefix = 'Progress:', suffix = 'Complete', length = 50)
        while i < len(data):
            if(len(batch) < BATCH_SIZE):
                batch.append(data[i].get("uri"))
                i += 1
                continue
            r = get_batch(session, batch)
            assert(r)
            batch = []
            for post in r:
                # Author contains DID so we no longer need it
                # rkey is also no longer relevant
                item = {
                    "uri": post.get("uri"),
                    "author": post.get("author"),
                    "record": post.get("record"),
                    "like_count": post.get("likeCount"),
                    "reply_count": post.get("replyCount"),
                    "repost_count": post.get("repostCount"),
                    "bookmark_count": post.get("bookmarkCount"),
                    "quote_count": post.get("quoteCount")
                }
                corpus.append(post)
                hydrated += 1
            print_progress(i, len(data) - 1, prefix = 'Progress:', suffix = 'Complete', length = 50)

        # Final batch if there's anything left
        if batch:
            r = get_batch(session, batch)
            assert(r)
            for post in r:
                item = {
                    "uri": post.get("uri"),
                    "author": post.get("author"),
                    "record": post.get("record"),
                    "like_count": post.get("likeCount"),
                    "reply_count": post.get("replyCount"),
                    "repost_count": post.get("repostCount"),
                    "bookmark_count": post.get("bookmarkCount"),
                    "quote_count": post.get("quoteCount")
                }
                corpus.append(post)
                hydrated += 1
            print_progress(len(data), len(data), prefix = 'Progress:', suffix = 'Complete', length = 50)
    return corpus, hydrated

if __name__ == "__main__":
    hydrated_corpus, hydrated = hydrate()
    new_json = {"total": hydrated, "data": hydrated_corpus}
    with DATA_PATH.open("r") as f:
        dry_corpus = json.load(f)
        total = dry_corpus.get("total")
    with OUT_PATH.open("w") as f:
        json.dump(new_json, f, indent = 4)
    print(f"{hydrated} data points hydrated and written to {OUT_PATH}")
    if (total - hydrated) > 0:
        print(f"{total - hydrated} URIs did not return a post and have been removed from the dataset.")