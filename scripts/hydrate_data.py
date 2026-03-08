"""
Hydrates ../data/corpus.json by scraping the AT API for more dynamic data such as like counts, and creates a new JSON at ../data/hydrated_corpus.json.
Run this script before using data to ensure that everything is up to date.
"""
import json, requests, time, argparse
from pathlib import Path
from bluesky_scraper import print_progress

URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts"
PROFILES_URL = "https://public.api.bsky.app/xrpc/app.bsky.actor.getProfiles"

DATA_DIR = Path("../data")
DATA_FILE = "corpus.json"
OUT_FILE = "hydrated_corpus.json"

DATA_PATH = DATA_DIR / DATA_FILE
OUT_PATH = DATA_DIR / OUT_FILE

POST_BATCH_SIZE = 25
PROFILE_BATCH_SIZE = 25

def has_media(post_item: dict) -> bool:
    media = post_item.get("media") or {}
    images = media.get("images") or []
    videos = media.get("videos") or []
    return bool(images or videos)

def extract_media_urls(post: dict) -> dict:
    """ 
    Extracts URLs from posts containing images or videos and returns the image or thumbnail

    Parameters:
        post - Required : the post to extract from (dict)
    """
    out = {"images": [], "videos": []}
    embeds = []
    if isinstance(post.get("embed"), dict):
        embeds.append(post["embed"])
    if isinstance(post.get("embeds"), list):
        embeds.extend([e for e in post["embeds"] if isinstance(e, dict)])

    def handle(e: dict):
        t = e.get("$type", "")

        # recordWithMedia nests the actual media under `media`
        if t in ("app.bsky.embed.recordWithMedia#view", "app.bsky.embed.recordWithMedia"):
            m = e.get("media")
            if isinstance(m, dict):
                handle(m)
            return

        if t == "app.bsky.embed.images#view":
            for img in e.get("images", []) or []:
                if isinstance(img, dict):
                    out["images"].append({
                        "thumb": img.get("thumb"),
                        "fullsize": img.get("fullsize"),
                        "alt": img.get("alt"),
                        "aspectRatio": img.get("aspectRatio"),
                    })
            return

        if t == "app.bsky.embed.video#view":
            out["videos"].append({
                "thumbnail": e.get("thumbnail"),
                "aspectRatio": e.get("aspectRatio"),
            })
            return
    
    for e in embeds:
        handle(e)
    return out


def get_batch(session: requests.Session, uris: list, retries = 4, timeout_s = 30, base_backoff = 0.8) -> list:
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

def get_profiles_batch(session: requests.Session, actors: list[str], retries: int = 4, timeout_s: int = 30, base_backoff: float = 0.8) -> list[dict]:
    """
    Requests the public Bluesky API for a batch of actor profiles.

    Uses repeated query params:
    ?actors=did1&actors=did2&...
    """
    params = [("actors", actor) for actor in actors]

    for attempt in range(retries + 1):
        try:
            resp = session.get(PROFILES_URL, params=params, timeout=timeout_s)

            if resp.status_code == 200:
                data = resp.json()
                return data.get("profiles", [])

            print(f"[getProfiles] status={resp.status_code}")
            print(f"[getProfiles] url={resp.url}")
            print(f"[getProfiles] body={resp.text[:500]}")

            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = float(retry_after)
                    except ValueError:
                        sleep_s = base_backoff * (2**attempt)
                else:
                    sleep_s = base_backoff * (2**attempt)

                if attempt < retries:
                    time.sleep(min(sleep_s, 30.0))
                    continue
                return []

            return []

        except requests.RequestException:
            if attempt < retries:
                time.sleep(min(base_backoff * (2**attempt), 30.0))
                continue
            return []
    return []

def build_post_item(post: dict) -> dict:
    """
    Converts a hydrated API post response into the normalized JSON structure.
    """
    record = post.get("record", {}) or {}
    reply = record.get("reply") or {}

    parent_uri = reply.get("parent", {}).get("uri")
    root_uri = reply.get("root", {}).get("uri")

    item = {
        "uri": post.get("uri"),
        "author": post.get("author"),
        "created_at": record.get("createdAt"),
        "text": record.get("text"),
        "reply": {
            "parent_uri": parent_uri,
            "root_uri": root_uri,
        },
        "media": extract_media_urls(post),
        "has_media": False,  # filled below
        "like_count": post.get("likeCount", 0),
        "reply_count": post.get("replyCount", 0),
        "repost_count": post.get("repostCount", 0),
        "bookmark_count": post.get("bookmarkCount", 0),
        "quote_count": post.get("quoteCount", 0),
    }
    item["has_media"] = has_media(item)
    return item


def hydrate_posts(source: Path) -> tuple[list[dict], int]:
    """
    Hydrates the raw corpus by querying post metadata from the public Bluesky API.
    """
    with source.open("r", encoding="utf-8") as f:
        dry_corpus = json.load(f)

    data = dry_corpus.get("data", [])
    corpus: list[dict] = []

    with requests.Session() as session:
        total = len(data)
        print_progress(0, max(total, 1), prefix="Posts:", suffix="Complete", length=50)

        for start in range(0, total, POST_BATCH_SIZE):
            batch_uris = [
                item.get("uri")
                for item in data[start : start + POST_BATCH_SIZE]
                if item.get("uri")
            ]

            posts = get_batch(session, batch_uris)

            # Skip failed batch instead of looping forever
            if posts:
                for post in posts:
                    corpus.append(build_post_item(post))

            print_progress(
                min(start + POST_BATCH_SIZE, total),
                max(total, 1),
                prefix="Posts:",
                suffix="Complete",
                length=50,
            )
    return corpus, len(corpus)

def hydrate_authors(hydrated_corpus: list[dict], batch_size: int = PROFILE_BATCH_SIZE) -> dict[str, dict]:
    """
    Queries public Bluesky profile stats for all unique authors in the corpus.
    """
    author_dids = sorted(
        {
            post.get("author", {}).get("did")
            for post in hydrated_corpus
                if post.get("author", {}).get("did")
        }
    )

    out: dict[str, dict] = {}

    if not author_dids:
        return out

    with requests.Session() as session:
        total = len(author_dids)
        print_progress(0, max(total, 1), prefix="Authors:", suffix="Complete", length=50)

        for start in range(0, total, batch_size):
            batch = author_dids[start : start + batch_size]
            profiles = get_profiles_batch(session, batch)

            for p in profiles:
                did = p.get("did")
                if not did:
                    continue

                out[did] = {
                    "did": did,
                    "handle": p.get("handle"),
                    "displayName": p.get("displayName"),
                    "followersCount": p.get("followersCount", 0) or 0,
                    "followsCount": p.get("followsCount", 0) or 0,
                    "postsCount": p.get("postsCount", 0) or 0,
                }

            print_progress(
                min(start + batch_size, total),
                max(total, 1),
                prefix="Authors:",
                suffix="Complete",
                length=50,
            )

    return out

def attach_author_data(posts: list[dict], scraped_author_stats: dict[str, dict]) -> None:
    """
    Attaches author profile stats and corpus-based priors to each post.
    """
    default_scraped = {
        "followersCount": 0,
        "followsCount": 0,
        "postsCount": 0,
    }

    for post in posts:
        did = post.get("author", {}).get("did")

        author_stats = scraped_author_stats.get(did, default_scraped)
        post["author_stats"] = {
            "followersCount": author_stats.get("followersCount", 0),
            "followsCount": author_stats.get("followsCount", 0),
            "postsCount": author_stats.get("postsCount", 0),
        }

def hydrate(source: Path) -> tuple[list[dict], int]:
    """
    Full hydration pipeline:
    1. Hydrate posts
    2. Hydrate authors
    3. Compute corpus-internal author stats
    4. Attach all author prior data to each post
    """
    hydrated_corpus, hydrated = hydrate_posts(source)

    scraped_author_stats = hydrate_authors(hydrated_corpus)

    attach_author_data(hydrated_corpus, scraped_author_stats=scraped_author_stats)

    return hydrated_corpus, hydrated

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hydrates data with Bluesky API")
    ap.add_argument("--in", dest="in_path", default=str(DATA_PATH), help="input path for corpus JSON (default: ../data/corpus.json)")
    ap.add_argument("--out", dest="out_path", default=str(OUT_PATH), help="output path for hydrated corpus JSON (default: ../data/hydrated_corpus.json)")
    args = ap.parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    hydrated_corpus, hydrated = hydrate(in_path)
    new_json = {"total": hydrated, "data": hydrated_corpus}
    with in_path.open("r") as f:
        dry_corpus = json.load(f)
        total = dry_corpus.get("total")
    with out_path.open("w") as f:
        json.dump(new_json, f, indent = 4)
    print(f"{hydrated} data points hydrated and written to {out_path}")
    if (total - hydrated) > 0:
        print(f"{total - hydrated} URIs did not return a post and have been removed from the dataset.")