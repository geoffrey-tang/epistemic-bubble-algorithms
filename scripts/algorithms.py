import json
import random
import math
import numpy as np
from pathlib import Path
from bluesky_scraper import print_progress
from datetime import datetime, timezone
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt


DATA_DIR = Path("../data")
GRAPH_DIR = Path("../data/graphs")
OUT_FILE = "algorithms_results.json"
IN_FILE = "hydrated_corpus_large.json"

OUT_PATH = DATA_DIR / OUT_FILE
IN_PATH = DATA_DIR / IN_FILE

def build_tfidf_matrix(posts, min_df=3, max_df=0.8, ngram_range=(1, 2)):
    """
    Builds vectorizer and matrix used for TF-IDF
    """
    texts = [p.get("text", "") or "" for p in posts]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def compute_total_author_engagement(posts):
    """
    Computes and returns a dict of the total amount of engagement of each author
    """
    totals = defaultdict(float)

    for p in posts:
        did = p["author"]["did"]

        likes = p.get("like_count", 0) or 0
        reposts = p.get("repost_count", 0) or 0
        replies = p.get("reply_count", 0) or 0
        quotes = p.get("quote_count", 0) or 0

        engagement = likes + 2*reposts + 1.5*replies + 2*quotes

        totals[did] += engagement

    return totals

def compute_author_priors(posts):
    """
    Calculates and returns a dict of scores regarding how much relevance an author has
    """
    engagement_totals = compute_total_author_engagement(posts)

    priors = {}

    for p in posts:
        did = p["author"]["did"]

        followers = p.get("author_stats", {}).get("followersCount", 0)
        posts_count = p.get("author_stats", {}).get("postsCount", 0)

        total_engagement = engagement_totals[did]

        prior = (
            0.5 * math.log1p(followers)
            + 0.3 * math.log1p(posts_count)
            + 0.2 * math.log1p(total_engagement)
        )

        priors[did] = prior

    return priors

def attach_author_prior(posts, priors):
    """
    Attaches the author prior to the corresponding authors in the dataset
    """
    for p in posts:
        did = p["author"]["did"]
        p["author_prior"] = priors.get(did, 0.0)

def parse_created_at(s: str) -> datetime:
    """
    Parses created_at time from an ISO format to a Python friendly format
    """
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)

def rank_chrono(posts, K: int):
    """
    Returns a list of K posts in chronological order
    """
    ranked = sorted(
        posts,
        key=lambda p: parse_created_at(p["created_at"]),
        reverse=True
    )
    return ranked[:K]

def rank_engagement(posts, K: int, tau_hours: float = 48.0):
    """
    Returns a list of K posts ranked by total engagement
    """
    def score(p):
        likes = p.get("like_count")
        reposts = p.get("repost_count")
        replies = p.get("reply_count")

        engagement = 1*likes + 2*reposts + 1.5*replies
        return engagement

    ranked = sorted(posts, key=score, reverse=True)
    return ranked[:K]

def rank_author_boost(posts, K: int, alpha: float = 1.0):
    """
    Returns a list of K posts ranked by total engagement, while artificially boosting popular authors
    """
    def score(p):
        likes = p.get("like_count", 0) or 0
        reposts = p.get("repost_count", 0) or 0
        replies = p.get("reply_count", 0) or 0
        quotes = p.get("quote_count", 0) or 0

        engagement = likes + 2 * reposts + 1.5 * replies + 2 * quotes

        author_prior = p.get("author_prior", 0.0)

        return engagement + alpha * author_prior

    ranked = sorted(posts, key=score, reverse=True)
    return ranked[:K]

def rank_tfidf_profile(posts, vectorizer, X, engaged_texts, K):
    """
    Builds TF-IDF vectors from a seed set of posts and returns a list of the K most similar posts
    """
    if not engaged_texts:
        return []

    # Convert engaged texts into TF-IDF vectors
    engaged_vecs = vectorizer.transform(engaged_texts)
    engaged_vecs = normalize(engaged_vecs)

    # Build user interest vector
    user_profile = np.asarray(engaged_vecs.mean(axis=0))
    user_profile = normalize(user_profile)

    # Compute similarity with all posts
    sims = cosine_similarity(user_profile, X).ravel()
    ranked_indices = sims.argsort()[::-1]

    engaged_set = set(engaged_texts)
    results = []
    for idx in ranked_indices:
        if posts[idx].get("text", "") in engaged_set:
            continue
        results.append(posts[idx])
        if len(results) >= K:
            break

    return results

def rank_random(posts, K, seed=0):
    """
    Returns a random list of K posts
    """
    random.seed(seed)
    return random.sample(posts, min(K, len(posts)))

def get_text_from_json(posts):
    """
    Extracts post text from a json
    """
    all_text = []
    for i in posts:
        text = i.get("text")
        all_text.append(text)
    return all_text

def build_topic_model() -> tuple[BERTopic, CountVectorizer, ClassTfidfTransformer]:
    """
    Build BERTopic model
    """
    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=True,
        reduce_frequent_words=True
    )
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        metric="cosine",
        random_state=42,
        min_dist=0.1
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1,2)
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        min_topic_size=10
    )
    return topic_model, vectorizer_model, ctfidf_model

def run_pipeline(sentences: list[str], feed_name: str, embedder: SentenceTransformer, out_dir: Path):
    """
    Runs a BERTopic pipeline on a list of strings, using both TF-IDF and embeddings to organize outliers
    """
    topic_model, vectorizer_model, ctfidf_model = build_topic_model()

    embeddings = embedder.encode(
        sentences,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    # running bertopic
    topics, probs = topic_model.fit_transform(sentences, embeddings)

    # cleaning up outliers using both c-TF-IDF and semantic embeddings
    if -1 in topics:
        topics = topic_model.reduce_outliers(
            sentences,
            topics,
            strategy="c-tf-idf"
        )
        topic_model.update_topics(sentences, topics=topics, vectorizer_model=vectorizer_model)
    if -1 in topics:
        topics = topic_model.reduce_outliers(
            sentences,
            topics,
            strategy="embeddings",
            embeddings=embeddings,
            threshold=0.2
        )
        topic_model.update_topics(sentences, topics=topics, vectorizer_model=vectorizer_model)
    topic_model.reduce_topics(sentences, nr_topics="auto")
    topics = topic_model.topics_
    info = topic_model.get_topic_info()

    # Save visualizations
    valid_topics = [t for t in info["Topic"].tolist() if t != -1]
    if len(valid_topics) >= 2:
        fig = topic_model.visualize_topics(topics=valid_topics, title=f"<b>{feed_name} Intertopic Distance Map</b>")
        fig.write_html(out_dir / f"{feed_name}_topics.html")

    fig2 = topic_model.visualize_documents(sentences, embeddings=embeddings, title=f"<b>{feed_name} Documents and Topics</b>")
    fig2.write_html(out_dir / f"{feed_name}_documents.html")

    return {
        "feed_name": feed_name,
        "topic_model": topic_model,
        "topics": topics,
        "probs": probs,
        "embeddings": embeddings,
        "topic_info": info,
        "representative_docs": {topic_id: topic_model.get_representative_docs(topic_id) for topic_id in info["Topic"]}
    }
    

if __name__ == "__main__":
    with IN_PATH.open("r") as f:
        corpus = json.load(f)
    data = corpus.get("data")
    vectorizer, X = build_tfidf_matrix(data)
    single_seed_set_politics = [
        "Pres. Reagan showing respect & tribute from the commander-in-chief to the fallen service members, attached a medal to the flag-draped caskets of the 4 Marines killed in El Salvador.\n\nTrump not very respectful wearing a stupid baseball cap as the 6 soldiers killed in his Iran war were returned home.",
        "The fact that even the Cheneys saw the danger of Trump to our country and certain voters (and non voters) didn't is the actual problem.",
        "Seeing Trump play commander in chief, we understand how Trump bankrupt a casino!#Trump",
        "Trump doesn't care about anyone. The only thing he cares about is how much money he can steal from America.",
        "When Fascist Career Criminals Steal Elections, We Get Crimes!!! Invoke Article ll, Section 4 Of The Constitution, Prosecute Trump, His Entire Administration, Most GOP Members Of Congress, All Conservative SCOTUS Justices, For Espionage And Treason!!!",
        "Stop campaigning against Dems every election and we wouldn't have had Trump. Blood is on your hands & that of your Blue MAGA cult members.",
        "The problem we're facing now is that there is no way to enforce the court orders that ICE consistently ignores. Trump has made it clear that he *wants* them to murder anyone they want, or the psychopaths who murdered Good and Pretti would already be behind bars.",
        "Holy Fucking Christ just how many Americans will have to die before they realize their President raped and strangled children, and Putin got it all on tape.\n\nTrump will NEVER harm Putin.\n\nTrump will KILL every single American before that happens.",
        "The Trump regime is using our laws against us.\n\nWe must flip this on them. \n\nWe have to get unprecedented.\n\nThey're not following the laws, but, our leadership is ignoring their lawlessness and following the law.\n\nIt's a death sentence for everyone.",
        "They are going to come after us all because of this demented trio: trump, miller, kegseth, and their cowardly enablers",
        "Trump calls them suckers and losers.  Meanwhile Russia helps Iran kill Americans, and Trump responds buy allowing them to sell oil and raise more money to fight against us.\n\nEvery last Republican enabler will answer for this in one way or another.",
        "This is truly sad, and unnecessary deaths, because of a pedo and a criminal to boot. And we call him Trump, and a selfish asshole who gets away with anything he wants. I can say this is for sure hell he isn't my President. Thanks a lot to the dumbasses who didn't vote.",
        "Trump's incompetence and outright corruption will be the stuff of political science case studies for decades to come.",
        "You didn't mention that classic abuser lie, in which the sadist doing the lying claims his victim actually thanked him for the pain. I grew up with someone like that. It's real. And Trump is increasingly indulging his thirst for others' pain - now on an INTERNATIONAL scale.",
        "our state leaders and law enforcement must work closer together across interstate lines to protect our citizens from terrorism caused by Trump's illegal war in Iran. Trump is trying to put people in harms way to grab power.",
    ]

    single_seed_set_tech = [
        "Aside from general Microsoft & Windows jank, which I could complain about all-day, I simply HATE Windows File Explorer with a burning passion.\n\nIt is... not very intuitive, it's slow as molasses & overall is clunky.\n\nI know many cannot use Linux & that is fine, but I can and greatly prefer it.",
        "normie-adjacent media outlets be like \"what Linux distro is right for you? Try SNOOFLE OS, the weirdo-ass distro out of some guy's basement who gets its firefox package updated once a year maybe if he gets around to it\"\n\nbased on an actual story of a mainstream-ish distro which I won't name",
        "going back from windows to linux always is just like a \"shit computers can be kinda fun actually\" moment\n\none of the things that constantly drove me insane on windows was the fact that just by having a hdd in my pc, explorer would fairly consistently hang for like 10 secs before opening",
        "LTT missed a good opportunity to have Wendell or Emily come in and explain/troubleshoot/give tips about linux filmed as a reaction after the fact.\n\nThat way we have the new user experience and an expert to help guide the audience. It would be a good resource for devs and new users.",
        "i installed a persistent kde plasma on my thumb drive and now i'm posting from that. it runs so much better than the windows 10 install in EVERY way. it would be nice to find out if one could get nvda working on linux with ease or not though (thats the screenreader the lioness uses)",
        "Think blocking cookies protects your privacy?\n\nNot really.\n\nBrowser fingerprinting can still identify you using your device, fonts, screen size, and more.\n\nI analyzed what my own browser reveals online.\n\nghostlyinc.com/en-us/my-onl...\n\n#privacy #security #webdev",
        "CVE-2026-3680 - RyuzakiShinji biome-mcp-server biome-mcp-server.ts command injection\nCVE ID : CVE-2026-3680\n \n Published : March 7, 2026, 11:15 p.m. | 1 hour, 56 minutes ago\n \n Description : A security flaw has been discovered in RyuzakiShinji biome-mcp-server up to 1.0.0. Aff...",
        "Wikipedia was affected by a self-replicating JavaScript worm that vandalized pages. This incident represents a significant cybersecurity challenge as the worm spread quickly across the platform.",
        "Cybersecurity is no longer a property that is inherent in hardware and software; it is a systemic property of organizations and, by extension, markets. The certification frameworks would be established by ENISA under the Commission\u2019s mandate, with periodic reviews at least every four years. (18/27)",
        "As a distraction from drafting, I sketched out an event bus device at /dev/events for my silly Inferno port to the CLR.\n\nI don't know if I'll ever use it, but the design is at least there.\n\nIt's partially inspired by netlink on Linux, just to bring the ioctl replacement obsession full-circle.",
        "So far so good with gettings things up and running.\n\nOnly quirk has been my microsoft bluetooth keyboard won't connect (the bastids!).\n\nInstalled Steam and currently downloading Returnal.\n\nEven got my 3d printing stuff up and running...\n\n... in Linux... \n\nIt's been easy but also tinkery.\n\n#linuxsky",
        "Ok last vent post about my PC and Linux situation lol.\nAfter fixing my string of Windows crashes, I decided to double-check my cooling parts' compatibility in liquidctl in case I try to jump ship to Linux again.\nI hadn't realized, not only are my fan controllers broken, but my liquid cooler too!",
        "I used CentOS at college and uni, then Ubuntu when I finally got a job in tech 12 years ago (which I hated and have never looked back). Arch linux was always the big scary linux and I didn't want anything to do with it but CachyOS has been so much more usable.",
        "it's BASIC, there's no compiling here- it's also BASIC as stand-in for an operating system so those vars could be something a .BAS file creates/calls- or they could be more like what we'd think of as bashlike environment variables. same pool for both",
        "Broadcom doesn't consider Microsoft an AI company? I mean, sure, Meta's years of making custom AI accelerators just ground to a halt but Broadcom is the company where hardware and software go to die",
        "I've found a method to get the \"tablet mode\" of the laptop, now I just need to be able to enable or disable the touchscreen (/dev/input/event8 on this laptop) via cli commands, & I can throw it into a systemd service to run at startup"
    ]
    
    mixed_seed_set = [
        # politics
        "Many of the Dems are corrupt too, and complicit with the fascists, including the party leadership. Their job is to feign opposition without really taking any meaningful action."
        "Imagine if those predicted terrorist attacks on US soil Trump treated w/cavalier disinterest first focused on Trump-owned offshore  properties. Think he'd still be disinterested? Imagine Lloyds or anyone else providing policies."
        "As long as regional newspapers & newspapers & news agencies affiliated with the Republican Party do not report on this support for Trump from the oligarchs Ellison, Zuckerberg, Bezos, Musk & Co. it is not real.#broligarchy #fascism #nepotism #plot #Iran",
        # art
        "We are an International Art Collective made out of 3 whimsy filled, queer fantasy artists!",
        "my brain has decided to just not understand how to draw today so i give up. been struggling with this pose for TOO long",
        "i've suffered years of psychic damage from comic book artists not knowing what clothes look like",
        "The Hollowed Beauty - Occtis Tachonis, the Necromancer Wizard - Is It Thursday Yet? Occtis, Occtis, Occtis, our sweet lil alt, sad victorian boy. He intrigues me, and I would do anything for Pin #CriticalRole #fanart #occtis #seekerstable",
        "Thank you for the artshare! Hope you find some cool artists to follow!",
        # pokemon
        "Haha, it can be done! Some of the pokemons and their quests are SO funny and adorable, you'll love those I think. Plus, if you go into this cave region you unlock the mermaid hair early on. Cuter than the other hair options!",
        "this has to be my favorite pokemon quote of all time. right after professor rowans rant @ team galactic that one time where he called them outlandish hooligans or whatever. I fucking love pokemon",
        "Mall of America was a fascinating place to Pokemon Go. Like 900 pokestops and 30 gyms. But man... this \"legendary pokemon are appearing in the wild/in raids, and you can't catch them no matter how many fruits and balls you throw at them\" is worse than not having ever seen them at all :p",
        "At some point Instagram realized I play Pokemon GO and my feed became one after another people getting rare event shinies and now I finally understand why Instagram causes 1,374 suicides per year",
        "Pewter City in #Pokopia, especially the night theme, has that sense of loneliness and sweet melancholy that I was desperately missing from ACNH. I sat still chilling with my pokemon multiple times during the night until I fell asleep.\n\nThis is the best game ever.",
        "i'm so glad i got this stupidly rare blind box magent that was only sold at the korean stop of the pokemon orchestra concert when i did because the few listings i see for it now are all $700-$900+ lmao",
        "My wife bought Pokemon Pokopia for our youngest kiddo. I decided to give it a try and suddenly the entire evening was gone and I'm crafting fire places for ghosts and busy planting a garden to make my newest pokemon happy.\n\nIt might be an ok game."
    ]

    feeds = [
        get_text_from_json(rank_chrono(data, 1000)),
        get_text_from_json(rank_random(data, 1000)),
        get_text_from_json(rank_engagement(data, 1000)),
        get_text_from_json(rank_author_boost(data, 1000)),
        get_text_from_json(rank_tfidf_profile(data, vectorizer, X, single_seed_set_politics, 1000)),
        get_text_from_json(rank_tfidf_profile(data, vectorizer, X, single_seed_set_tech, 1000)),
        get_text_from_json(rank_tfidf_profile(data, vectorizer, X, mixed_seed_set, 1000))
    ]
    names = [
        "Chronological",
        "Random",
        "Engagement",
        "Engagement-Author-Boost",
        "Similarity-Politics",
        "Similarity-FOSS-Tech",
        "Similarity-Mixed"
    ]

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    for i in range(len(feeds)):
        run_pipeline(feeds[i], names[i], embedder, GRAPH_DIR)
    
