# Podcast Recommendation System

This is a hobby project which would recommend podcasts based on their similarity
to another podcast using a Vector database.

It uses a HuggingFace wrapper arour Fairseq S2T model to extract the transcript
of the podcast and then indexes it using MiniLM-L6-V2 SentenceTransformers as
a vectorizer.

Both models are really small which allows running at decent speed on a fairly
old CPU (my dev machine uses a intel i7 6-th gen). Accuracy could certainly be
improved with larger, better models, should a little extra compute power
be available, especially when it comes to the vectorization (overall we don't
care if the transcription is not perfect as no human will ever read it anyway).
Either way this is merely a POC and this is good enough for now.


## How to run

* install [docker](https://docs.docker.com/engine/install/ubuntu/)
* install [docker compose](https://docs.docker.com/compose/install/linux/)
* install the python requirements: `pip3 install -r requirements.txt`.
* start the milvus db with `./start_db.sh`
* Run the dummy db test `python3 test_db.py` to ensure everything is
up and running.
* You can now start indexing using
    `python3 index_podcasts.py <number_of_new_podcasts_to_index>`. In a
    real app, you would run something like this asynchronously, listening for
    new podcasts being released from some API.
* You can get recommandations for a given podcast using
    `python3 get_recommandation.py <url-or-path>`. Again in a real app, this
    would run in the background, and be triggered by some RESTful API for
    instance.


## Indexing new podcasts

This demo fetches podcasts from
[the PodcastIndex.org](https://github.com/Podcastindex-org) for simplicity.
