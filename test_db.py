import csv
import os
import shutil
import time

import gdown
from loguru import logger
import pymilvus
import zipfile

import vector_db

def test_text_vector_db():
    logger.info('Downloading test data...')
    url = 'https://drive.google.com/uc?id=11ISS45aO2ubNCGaC3Lvd3D7NT8Y7MeO8'
    output = './movies.zip'
    gdown.download(url, output)

    with zipfile.ZipFile("./movies.zip","r") as zip_ref:
        zip_ref.extractall("./movies")

    COLLECTION_NAME = 'test_db'
    pymilvus.connections.connect(host='localhost', port=str(19530))
    if pymilvus.utility.has_collection(COLLECTION_NAME):
        pymilvus.utility.drop_collection(COLLECTION_NAME)

    record = []
    with open('./movies/plots.csv', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i > 1000:
                break
            if '' in (row[1], row[7]):
                continue
            record.append(vector_db.DbEntry(description=row[1], text=row[7]))

    logger.info('Indexing data in DB...')
    db = vector_db.TextVectorDb(name=COLLECTION_NAME)
    start = time.perf_counter()
    db.batch_insert(record)
    logger.info(f'Indexed 1000 examples in {time.perf_counter() - start:.2f} '
                'seconds')

    logger.info('Querying DB for closest plot match...')
    search_terms = ['A movie about cars', 'A movie about monsters']
    start = time.perf_counter()
    results = db.batch_query(search_terms)

    logger.info(f'Querried 2 inputs in {time.perf_counter() - start:.2f} '
                'seconds')

    for term, hits in zip(search_terms, results):
        hits_titles = [hit.entity.get(vector_db.DbKey.description.value)
                       for hit in hits]
        logger.info(f"Results for {term}: {hits_titles}")

    pymilvus.utility.drop_collection(COLLECTION_NAME)
    shutil.rmtree('./movies.zip', ignore_errors=True)
    os.remove('./movies')


if __name__ == "__main__":
    test_text_vector_db()
