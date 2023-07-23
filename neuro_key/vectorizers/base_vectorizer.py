from abc import ABC, abstractmethod
from typing import List, Union

from pydantic.dataclasses import dataclass

from neuro_key.data import Document
from neuro_key.helpers import gpu_util


@dataclass
class BaseVectorizerConfig:
    """Base config for vectorizers.

    Args:
        device: The device to use for embedding.
        model_name_or_path: The path to the model.
        batch_size: The batch size to use when embedding a batch of documents.
        content_field: The field of the document to embed.
        show_progress_bar: Whether to show a progress bar when embedding a batch of documents.
    """

    model_name_or_path: str
    device: str = "auto"
    batch_size: int = 8
    content_field: str = "text"
    embedding_field: str = "embedding"
    show_progress_bar: bool = False


class BaseVectorizer(ABC):
    def __init__(self, config: BaseVectorizerConfig):
        """
        auto: choose gpu if present else use cpu
        cpu: use cpu
        cuda:{id} - cuda device id
        """
        self.config = config
        _device_id = config.device
        self._device: int = gpu_util.get_device_id(_device_id)
        self._model_name_or_path: str = config.model_name_or_path
        self._batch_size = config.batch_size
        self._content_field = config.content_field
        self._show_progress_bar = config.show_progress_bar
        self._embedding_field = config.embedding_field

    @abstractmethod
    def _process(self, document: Document, **kwargs) -> Document:
        """Embed a single document.

        Args:
            document: A single document.

        Returns:
            A single document with embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def _process_batch(self, documents: List[Document], **kwargs) -> List[Document]:
        """Embed a batch of documents.

        Args:
            documents: A batch of documents.

        Returns:
            A batch of documents with embeddings.
        """
        raise NotImplementedError

    def run(self, documents: Union[Document, List[Document]], **kwargs) -> List[Document]:
        """Embed a single document or a batch of documents using sent2vec.

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
