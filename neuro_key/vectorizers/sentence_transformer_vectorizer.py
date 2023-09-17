from typing import List

from pydantic.dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from neuro_key.data import Document
from neuro_key.vectorizers import BaseVectorizer, BaseVectorizerConfig


@dataclass
class SentenceTransformerVectorizerConfig(BaseVectorizerConfig):
    ...


class SentenceTransformerVectorizer(BaseVectorizer):
    """Embed a document or a batch of documents using SentenceTransformer.

    Args:
        config: A dictionary containing the following keys:
            model_name_or_path: The path to the SentenceTransformer model.
    """

    def __init__(self, config: SentenceTransformerVectorizerConfig):
        super().__init__(config)

        self._model = SentenceTransformer(self._model_name_or_path)

    def _process(self, document: Document, **kwargs) -> Document:
        """Embed a single document using SentenceTransformer.

        Args:
            document: A single document.

        Returns:
            A single document with SentenceTransformer embeddings.
        """
        content = getattr(document, self._content_field)
        setattr(document, self._embedding_field, self._model.encode([content])[0])
        return document

    def _process_batch(self, documents: List[Document], **kwargs) -> List[Document]:
        """Embed a batch of documents using SentenceTransformer.

        Args:
            documents: A batch of documents.

        Returns:
            A batch of documents with with SentenceTransformer embeddings.
        """
        if all(isinstance(getattr(document, self._content_field), List) for document in documents):
            embeddings = self._model.encode(
                [item for document in documents for item in getattr(document, self._content_field)],
                show_progress_bar=self._show_progress_bar,
            )
            for idx, _ in enumerate(documents):
                if idx == 0:
                    start_idx = idx
                else:
                    start_idx = len(getattr(documents[idx - 1], self._content_field))
                end_idx = start_idx + len(getattr(documents[idx], self._content_field))
                setattr(documents[idx], self._embedding_field, embeddings[start_idx:end_idx])
        else:
            for idx, embedding in enumerate(
                self._model.encode(
                    [getattr(document, self._content_field) for document in documents], show_progress_bar=self._show_progress_bar
                )
            ):
                setattr(documents[idx], self._embedding_field, embedding)

        return documents
