"""
Reads in data from Bluesky's Jetstream as a reservoir sample, and collects the relevant data into ../data/corpus.json.
Collected data needs to be hydrated with hydrate_data.py before use.
"""
import asyncio, json, random, time, websockets, argparse
from pathlib import Path

JETSTREAM = "wss://jetstream1.us-west.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"
DATA_DIR = Path("../data")
OUT_FILE = "corpus.json"

OUT_PATH = DATA_DIR / OUT_FILE

SAMPLE_SIZE = 100
DURATION_S = 30

# Print iterations progress
def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters

    Parameters:
        iteration- Required : current iteration (int)
        total    - Required : total iterations (int)
        prefix   - Optional : prefix string (str)
        suffix   - Optional : suffix string (str)
        decimals - Optional : positive number of decimals in percent complete (int)
        length   - Optional : character length of bar (int)
        fill     - Optional : bar fill character (str)
        printEnd - Optional : end character (e.g. "\r", "\r\n") (str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

async def reservoir_sample(source, duration_s, k):
    """ 
    Reservoir sampling K items from Bluesky jetstream
    Ensures that every post we sample has a K/N chance of being in the dataset, no matter what N is, where:
        N = total seen posts
        K = number of posts included in sample

    Parameters:
        source     - Required : Jetstream instance to target (str)
        duration_s - Required : Duration in seconds to scrape (int)
        k          - Required : Sample size (int)
    """
    sample = []
    seen_uris = set()
    n = 0
    timer_start = time.time()

    print_progress(0, duration_s, prefix = 'Progress:', suffix = 'Complete', length = 50)

    async with websockets.connect(source, ping_interval=20, ping_timeout=20) as ws:
        while time.time() - timer_start < duration_s:
            print_progress(int(time.time() - timer_start), duration_s, prefix = 'Progress:', suffix = 'Complete', length = 50)
            # Get post
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                continue

            # Load JSON
            evt = json.loads(msg)
            commit = evt.get("commit", {})
            record = commit.get("record")

            # Validate for English text posts
            if not commit:
                continue
            if commit.get("operation") != "create":
                continue
            if record.get("text") is None or record.get("text") == "":
                continue
            if record.get("langs") is None:
                continue
            if "en" not in record.get("langs"):
                continue

            # Generate URI and deduplicate
            did = evt.get("did")
            collection = commit.get("collection")
            rkey = commit.get("rkey")
            uri = "at://" + did + "/" + collection + "/" + rkey
            if uri in seen_uris:
                continue
            seen_uris.add(uri)

            n += 1
            item = {
                "uri": uri,
                "did": did,
                "rkey": rkey,
                "record": record
            }
            # Reservoir sample
            if len(sample) < k:
                sample.append(item)
            else:
                pick_random = random.randint(0, n - 1)
                if pick_random < k:
                    sample[pick_random] = item             
    print_progress(duration_s, duration_s, prefix = 'Progress:', suffix = 'Complete', length = 50)
    print(f"Collected {len(sample)} posts from {n} total seen.")
    return sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reads in data from Bluesky's Jetstream as a reservoir sample, and collects the relevant data into ../data/corpus.json.\nCollected data needs to be hydrated with hydrate_data.py before use.")
    parser.add_argument("duration_s", type=int, help="Duration in seconds to scrape for")
    parser.add_argument("sample_size", type=int, help="Target number of posts to collect")
    args = parser.parse_args()

    corpus = asyncio.run(reservoir_sample(JETSTREAM, args.duration_s, args.sample_size))
    new_json = {"total": args.sample_size, "data": corpus}
    with OUT_PATH.open("w") as f:
        json.dump(new_json, f, indent = 4)
    print(f"Data written to {OUT_PATH}")