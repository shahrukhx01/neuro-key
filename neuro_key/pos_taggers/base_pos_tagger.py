import re
from abc import ABC, abstractmethod
from typing import List, Set, Union

import nltk
from pydantic.dataclasses import dataclass

from neuro_key.data import Document
from neuro_key.helpers import POS_GRAMMAR, SupportedLanguages, gpu_util


@dataclass
class BasePOSTaggerConfig:
    """Base config for POS taggers.

    Args:
        model_name_or_path: The name or path to the model.
        content_field: The field of the document to embed.
        device: The device to use for model.
        separator: The separator to use when joining the tokens and tags.
        allow_overlapping_ngrams: Whether to allow overlapping ngrams.
        max_keyphrase_words: The maximum number of words in a keyphrase.
    """

    model_name_or_path: str
    content_field: str = "text"
    device: str = "auto"
    separator: str = "|"
    considered_tags = {"NN", "NNS", "NNP", "NNPS", "JJ"}
    allow_overlapping_ngrams: bool = False
    max_keyphrase_words: int = 5


class BasePOSTagger(ABC):
    def __init__(self, config: BasePOSTaggerConfig):
        """
        auto: choose gpu if present else use cpu
        cpu: use cpu
        cuda:{id} - cuda device id
        """
        self.config = config
        _device_id = config.device
        self._device: int = gpu_util.get_device_id(_device_id)
        self._model_name_or_path: str = config.model_name_or_path
        self._considered_tags = config.considered_tags
        self._content_field = config.content_field
        self._separator = config.separator
        self._allow_overlapping_ngrams = config.allow_overlapping_ngrams
        self._max_keyphrase_words = config.max_keyphrase_words

    @staticmethod
    def _get_pos_grammar(language: SupportedLanguages):
        if language and language not in POS_GRAMMAR:
            raise ValueError(f"Language {language} is not supported.")

        return POS_GRAMMAR[language]

    def _extract_keyphrase_candidates(self, document: Document) -> Document:
        """
        Extract keyphrase candidates from a document.

        Args:
            document: A single document.

        Returns:
            A list of keyphrase candidates.
        """

        def _unique_ngram_candidates(texts: List[str]) -> List[str]:
            """
            Remove keyphrase ngrams that are subset of other ngrams.

            Args:
                texts: list of strings

            Returns:
                List of strings with non overlapping ngrams.
                Example:
                    Input: ['machine learning', 'machine', 'backward induction', 'induction', 'start']
                    Output: ['backward induction', 'start', 'machine learning']
            """
            results: List[str] = []
            for text in sorted(set(texts), key=len, reverse=True):
                if not any(re.search(r"\b{}\b".format(re.escape(text)), result) for result in results):
                    results.append(text)
            return results

        keyphrase_candidate: Set[str] = set()

        if not document.language:
            raise ValueError(f"Language is not set for document with Id {document.id}.")

        noun_phrase_parser = nltk.RegexpParser(self._get_pos_grammar(document.language))
        trees = noun_phrase_parser.parse_sents(document.pos_tags)  # generator with one tree per sentence

        for tree in trees:
            for subtree in tree.subtrees(filter=lambda subtree: subtree.label() == "NP"):  # for each nounphrase
                # concatenate the token with a space
                keyphrase_candidate.add(" ".join(word for word, tag in subtree.leaves()))

        keyphrase_candidates = [
            keyphrase for keyphrase in keyphrase_candidate if len(keyphrase.split()) <= self._max_keyphrase_words
        ]

        if self._allow_overlapping_ngrams:
            keyphrase_candidates = list(keyphrase_candidates)
        else:
            keyphrase_candidates = _unique_ngram_candidates(keyphrase_candidates)

        document.keyphrase_candidates = keyphrase_candidates
        return document

    @abstractmethod
    def _get_pos_tags(self, document: Document) -> Document:
        """Get POS tags for a document.

        Args:
            document: A single document.

        Returns:
            A single document with POS tags.
        """
        raise NotImplementedError

    def _process(self, document: Document, **kwargs) -> Document:
        """Process a single document.

        Args:
            document: A single document.

        Returns:
            A single document with POS tags.
        """
        return self._get_pos_tags(document)

    def _process_batch(self, documents: List[Document], **kwargs) -> List[Document]:
        """Process a batch of documents.

        Args:
            documents: A list of documents.

        Returns:
            A list of documents with POS tags.
        """
        return [self._get_pos_tags(document) for document in documents]

    def run(self, documents: Union[Document, List[Document]], **kwargs) -> List[Document]:
        """Run the POS tagger on a single document or a batch of documents.

        Args:
            documents: A single document or a list of documents.

        Returns:
            A single document or a list of documents with POS tags.
        """
        if isinstance(documents, Document):
            return self._process(document=documents)
        elif isinstance(documents, List) and all(isinstance(document, Document) for document in documents):
            return self._process_batch(documents=documents)

        raise TypeError(f"Type {type(documents)} is not supported by {self.__class__.__name__}")
