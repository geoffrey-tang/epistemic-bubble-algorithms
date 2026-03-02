import json
import random
import math
from pathlib import Path
from bluesky_scraper import print_progress
from datetime import datetime, timezone


DATA_DIR = Path("../data")
OUT_FILE = "algorithms_results.json"
IN_FILE = "hydrated_corpus.json"

OUT_PATH = DATA_DIR / OUT_FILE
IN_PATH = DATA_DIR / IN_FILE

def parse_created_at(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)

def rank_chrono(posts, K: int):
    ranked = sorted(
        posts,
        key=lambda p: parse_created_at(p["created_at"]),
        reverse=True
    )
    return ranked[:K]

def rank_engagement(posts, K: int, tau_hours: float = 48.0):
    def score(p):
        likes = p.get("like_count")
        reposts = p.get("repost_count")
        replies = p.get("reply_count")

        engagement = 1*likes + 2*reposts + 1.5*replies
        return engagement

    ranked = sorted(posts, key=score, reverse=True)
    return ranked[:K]

def rank_random(posts, K, seed=0):
    random.seed(seed)
    return random.sample(posts, min(K, len(posts)))

if __name__ == "__main__":
    with IN_PATH.open("r") as f:
        corpus = json.load(f)
    data = corpus.get("data")
    chron = rank_engagement(data, 1000)
    with OUT_PATH.open("w") as f:
        json.dump(chron, f, indent=4)
