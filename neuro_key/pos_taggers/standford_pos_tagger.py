from typing import List, Union

from pydantic.dataclasses import dataclass

from neuro_key.data import Document
from neuro_key.pos_taggers import BasePOSTagger, BasePOSTaggerConfig


@dataclass
class StanfordPOSTaggerConfig(BasePOSTaggerConfig):
    ...


class StanfordPOSTagger(BasePOSTagger):
    """ """

    def __init__(self, config: StanfordPOSTaggerConfig):
        super().__init__(config)
        ...

    def _process(self, document: Document, **kwargs) -> Document:
        ...
        return document

    def _process_batch(self, documents: List[Document], **kwargs) -> List[Document]:
        ...
        return documents

    def run(self, documents: Union[Document, List[Document]], **kwargs) -> List[Document]:
        ...
        return documents
