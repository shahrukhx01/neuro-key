from abc import ABC, abstractmethod
from typing import List, Union

from pydantic.dataclasses import dataclass

from neuro_key.data.document import Document


@dataclass
class BaseRankerConfig:
    ...


class BaseRanker(ABC):
    def __init__(self, config: BaseRankerConfig):
        self.config = config

    @abstractmethod
    def _process(self, document: Document, **kwargs) -> Document:
        """Rank a single document.

        Args:
            document: A single document.

        Returns:
            A single document with embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def _process_batch(self, documents: List[Document], **kwargs) -> List[Document]:
        """Rank a batch of documents.

        Args:
            documents: A batch of documents.

        Returns:
            A batch of documents with embeddings.
        """
        raise NotImplementedError

    def run(self, documents: Union[Document, List[Document]], **kwargs) -> List[Document]:
        """Rank a single document or a batch of documents using sent2vec.

        Args:
            documents: A single document or a batch of documents.

        Returns:
            A single document or a batch of documents with with sent2vec embeddings.
        """
        if isinstance(documents, Document):
            return self._process(document=documents)
        elif isinstance(documents, List) and all(isinstance(document, Document) for document in documents):
            return self._process_batch(documents=documents)

        raise TypeError(f"Type {type(documents)} is not supported by {self.__class__.__name__}")
