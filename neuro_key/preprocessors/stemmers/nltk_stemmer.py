from enum import Enum
from typing import List, Union

from nltk.stem import PorterStemmer, SnowballStemmer
from pydantic.dataclasses import dataclass

from neuro_key.data.document import Document
from neuro_key.preprocessors.stemmers.base_stemmer import BaseStemmer, BaseStemmerConfig


@dataclass
class NLTKStemmerConfig(BaseStemmerConfig):
    stemmer_type: str
    language: str


class NLTKStemmerType(str, Enum):
    PORTER_STEMMER = PorterStemmer
    SNOWBALL_STEMMER = SnowballStemmer


class LanguageMap(str, Enum):
    EN = "english"
    DE = "german"


class NLTKStemmer(BaseStemmer):
    def __init__(self, config: NLTKStemmerConfig):
        super().__init__(config)
        language = config.language
        self._stemmer: Union[PorterStemmer, SnowballStemmer] = NLTKStemmerType[config.stemmer_type](
            language=LanguageMap[language]
        )  # type: ignore

    def _stem(self, document: Document, **kwargs) -> Document:
        """Stem a single document's keyphrase candidates.

        Args:
            document: A single document with keyphrase candidates.

        Returns:
            A single document with stemmed keyphrase candidates.
        """
        content: Union[str, List[str]] = getattr(document, self._content_field)
        if isinstance(content, str):
            content = content.split()
        document.keyphrase_candidates = [self._stemmer.stem(token) for token in content]
        return document

    def _process(self, document: Document, **kwargs) -> Document:
        """Stem a single document's keyphrase candidates.

        Args:
            document: A single document with keyphrase candidates.

        Returns:
            A single document with stemmed keyphrase candidates.
        """
        return self._stem(document)

    def _process_batch(self, documents: List[Document], **kwargs) -> List[Document]:
        """Stem a batch of documents' keyphrase candidates.

        Args:
            documents: A batch of documents.

        Returns:
            A batch of documents with stemmed keyphrase candidates.
        """
        return [self._stem(document) for document in documents]
