import dataclasses
import enum
import itertools
from typing import Generator, Iterable, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pymilvus
from sentence_transformers import SentenceTransformer

import speech2text


@dataclasses.dataclass
class DbEntry:
    description: str
    text: str


@dataclasses.dataclass
class SpeechAudioEntry:
    audio_source: str
    transcript: Optional[str] = None

    def to_db_entry(self) -> DbEntry:
        if self.transcript is None:
            raise ValueError('Cannot convert to DbEntry without transcript')
        return DbEntry(description=self.audio_source,
                       text=self.transcript)


@dataclasses.dataclass
class AudioFile:
    audio: np.ndarray
    sample_rate: int


T = TypeVar('T')


def batched(iterable: Iterable[T], n) -> Generator[Tuple[T], None, None]:
    """Batch data into tuples of length n. The last batch may be shorter.

    Taken from the itertools recipes:
    https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


class DbKey(str, enum.Enum):
    id ='id'
    description = 'description'
    embedding = 'embedding'

class TextVectorDb:
    def __init__(self,
                 *,
                 name: str,
                 embedding_size: int = 384,
                 db_url: str = 'localhost',
                 port: int = 19530,
                 max_batchsize: int = 128) -> None:

        self._max_batchsize = max_batchsize

        pymilvus.connections.connect(host=db_url, port=str(port))
        self._schema = self._init_schema(embedding_size)
        self._collection = self._init_collection(name, self._schema)
        self._vectorizer: SentenceTransformer = None

    def _init_schema(self, embedding_size: int) -> pymilvus.CollectionSchema:
        fields = [
            pymilvus.FieldSchema(name=DbKey.id.value,
                                 dtype=pymilvus.DataType.INT64,
                                 is_primary=True,
                                 auto_id=True),
            pymilvus.FieldSchema(name=DbKey.description.value,
                                 dtype=pymilvus.DataType.VARCHAR,
                                 max_length=200),
            pymilvus.FieldSchema(name=DbKey.embedding.value,
                                 dtype=pymilvus.DataType.FLOAT_VECTOR,
                                dim=embedding_size)
        ]
        return pymilvus.CollectionSchema(fields=fields)

    def _init_collection(self,
                         name: str,
                         schema: pymilvus.CollectionSchema
                         ) -> pymilvus.Collection:
        collection = pymilvus.Collection(name=name, schema=schema)
        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{'nlist': 1536}
        }
        collection.create_index(field_name=DbKey.embedding.value,
                                index_params=index_params)
        collection.load()
        return collection

    @property
    def vectorizer(self):
        if self._vectorizer is None:
            self._vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
        return self._vectorizer

    def batch_insert(self, entries: Sequence[DbEntry]):
        """Inserts several entries into the database at once.

        Args:
            entries (Sequence[DbEntry]): Entries to insert.
        """
        for batch in batched(entries, self._max_batchsize):
            text_input = [entry.text for entry in batch]
            vector_representation = self.vectorizer.encode(text_input)
            self._collection.insert([
                [entry.description for entry in batch],
                [representation for representation in vector_representation]
            ])
        self._collection.flush()

    def insert_one(self, entry: DbEntry):
        """Inserts a single entry into the database.

        Args:
            entry (DbEntry): Entry to insert.
        """
        self.batch_insert([entry])

    def batch_query(self,
                    texts: Sequence[str],
                    top_k: int = 3) -> pymilvus.SearchResult:
        """Queries the db for the most similar entries to some input texts.

        Note: If needed, this could be made asynchronous as sollections search
        can return a Future.

        Args:
            texts (Sequence[str]): A number of input texts to query similar
                entries for.
            top_k (int): Number of hits to return per input. Defaults to 3.

        Returns:
            pymilvus.SearchResult: The closest hits for each input text
        """
        search_data = self.vectorizer.encode(texts)

        return self._collection.search(
            data=[representation for representation in search_data],
            anns_field=DbKey.embedding.value,
            param={},
            limit=top_k,
            output_fields= [DbKey.description.value]
        )


    def query_one(self, text: str, top_k: int = 3) -> pymilvus.Hits:
        """Queries the database for the top_k most similar entries to some text.

        Args:
            text (str): Input text
            top_k (int): Number of documents to query. Defaults to 3.

        Returns:
            pymilvus.Hits: Closests hits.
        """
        return self.batch_query([text], top_k)[0]


class TranscriptDb(TextVectorDb):
    def __init__(self, **kwargs) -> None:
        if 'name' not in kwargs:
            kwargs['name'] = 'transcripts_db'
        super().__init__(**kwargs)
        self._transcriber = None

    @property
    def transcriber(self):
        if self._transcriber is None:
            self.transcriber = speech2text.Speech2Text()
        return self._transcriber

    def batch_transcribe(self, entries: Sequence[SpeechAudioEntry]):
        return super().batch_insert(entries)

    def insert_one_from_audio(self, entry: SpeechAudioEntry):
        """Inserts a single entry into the database, directly as audio.

        Args:
            entry (SpeechAudioEntry): Entry to insert.
        """
        self.batch_insert_from_audio([entry])

    def load_audio(self, source: str) -> AudioFile:
        raise NotImplementedError('Audio loading not implemented.')

    def batch_transcript(self,
                         entries: Sequence[SpeechAudioEntry]
                        ) -> Sequence[SpeechAudioEntry]:
        """Transcribes a batch of entries.

        TODO: Actually batch this.

        Args:
            entries (Sequence[SpeechAudioEntry]): Entries to transcribe.

        Returns:
            Sequence[SpeechAudioEntry]: Transcribed entries.
        """
        for entry in entries:
            audio = self.load_audio(entry.audio_source)
            entry.transcript = self.transcriber(audio)

        return entries

    def transcript_one(self, entry: SpeechAudioEntry) -> SpeechAudioEntry:
        """Transcribes a single entry.

        Args:
            entry (SpeechAudioEntry): Entry to transcribe.
        """
        return self.batch_transcript([entry])[0]

    def batch_insert_from_audio(self, entries: Sequence[SpeechAudioEntry]):
        """Inserts several entries into the database at once, directly as audio.

        Args:
            entries (Sequence[SpeechAudioEntry]): Entries to insert.
        """
        entries = self.batch_transcript(entries)
        self.batch_insert([entry.to_db_entry() for entry in entries])

    def batch_query_from_audio(self,
                               audio_sources: Sequence[SpeechAudioEntry]
                              ) -> pymilvus.SearchResult:
        """Queries the db for the most similar entries to some input audios.

        Args:
            audios (Sequence[SpeechAudioEntry]): A number of input audio clips
                to query similar entries for. These should be voice recordings.

        Returns:
            pymilvus.SearchResult: The closest hits for each input text
        """
        audio_sources = self.batch_transcribe(audio_sources)
        return self.batch_query([entry.transcript for entry in audio_sources])
