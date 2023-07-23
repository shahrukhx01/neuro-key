from typing import List, Optional, Tuple, Union

from numpy.typing import NDArray
from pydantic import BaseModel

from neuro_key.helpers import SupportedLanguages


class Document(BaseModel):
    """Dataclass for a single document.

    Args:
        id: The id of the document.
        text: The text of the document.
        language: The language of the document.
        label: The label of the document.
        embedding: The embedding of the document.
        pos_tags: The pos tags of the document.
        keyphrase_candidates: The keyphrase candidates of the document.
        keyphrase_candidates_embeddings: Sentence embeddings of the keyphrase candidates.
    """

    id: str
    text: str
    language: Optional[SupportedLanguages]
    embedding: Optional[Union[List[float], NDArray]] = None
    pos_tags: Optional[List[List[Tuple[str, str]]]] = None
    keyphrase_candidates: Optional[List[str]] = None
    keyphrase_candidates_embeddings: Optional[List[Union[List[float], NDArray]]] = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
