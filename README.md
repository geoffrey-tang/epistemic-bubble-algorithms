# epistemic-bubble-algorithms
An analysis of the relation between social media algorithms, and epistemic bubbles and hermeneutical injustice using Bluesky's public API.

# Creating the virtual environment
In the epistemic-bubble-algorithms directory:
```
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

Once the virtual environment is created, you will only need to run
``` .venv\Scripts\activate ```
whenever you want to activate the virtual environment later.

# Directory Structure

## scripts
This folder contains the scripts used to gather data and run experiments.

***bluesky_scraper.py*** gathers posts via Bluesky's Jetstream API, and takes a reservoir sample. 

***hydrate_data.py*** gathers metadata via Bluesky's public API and creates a new corpus with the hydrated metadata.

***algorithms.py*** runs an algorithm suite on a given corpus, and uses BERTopic to create visualizations of topic clusters.
The algorithm suite contains the following:
```
Chronological
Random
Engagement
Engagement (Author Boosted)
TF-IDF
```

## data
Contains raw data and interactive visualizations as HTML files, which can be loaded in the browser.

# View specific visualizations
I ran out of time to make this pretty so you get to click a bunch of links

[Chronological Scatter Plot](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Chronological_documents.html)

[Chronological Intertopic Distance Map](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Chronological_topics.html)

[Engagement (Author Boosted) Scatter Plot](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Engagement-Author-Boost_documents.html)

[Engagement (Author Boosted) Intertopic Distance Map](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Engagement-Author-Boost_topics.html)

[Engagement Scatter Plot](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Engagement_documents.html)

[Engagement Intertopic Distance Map](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Engagement_topics.html)

[Random Scatter Plot](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Random_documents.html)

[Random Intertopic Distance Map](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Random_topics.html)

[TF-IDF (Tech) Scatter Plot](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Similarity-FOSS-Tech_documents.html)

[TF-IDF (Tech) Intertopic Distance Map](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Similarity-FOSS-Tech_topics.html)

[TF-IDF (Mixed) Scatter Plot](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Similarity-Mixed_documents.html)

[TF-IDF (Mixed) Intertopic Distance Map](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Similarity-Mixed_topics.html)

[TF-IDF (Politics) Scatter Plot](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Similarity-Politics_documents.html)

[TF-IDF (Politics) Intertopic Distance Map](https://geoffrey-tang.github.io/epistemic-bubble-algorithms/data/graphs/Similarity-Politics_topics.html)
