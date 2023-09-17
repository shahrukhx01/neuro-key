import spacy
from loguru import logger
from pydantic.dataclasses import dataclass

from neuro_key.data import Document
from neuro_key.pos_taggers import BasePOSTagger, BasePOSTaggerConfig


@dataclass
class SpacyPOSTaggerConfig(BasePOSTaggerConfig):
    ...


class SpacyPOSTagger(BasePOSTagger):
    """Spacy POS Tagger for tagging tokens with Parts-of-Speech (POS) tags.

    Args:
        block_config (Dict): Block configuration.
        **kwargs (Any): Keyword arguments.
    """

    def __init__(self, config: SpacyPOSTaggerConfig):
        super().__init__(config)
        try:
            self._model = spacy.load(self._model_name_or_path)
        except OSError:
            logger.error(
                f"Spacy can't find model '{self._model_name_or_path}'.\nTo install it run:\n\n`python -m spacy download"
                f" {self._model_name_or_path}`"
            )

    def _get_pos_tags(self, document: Document) -> Document:
        """Perform POS tagging on a single document.

        Args:
            document (Document): Document to be POS tagged.

        Returns:
            Document: Document with POS tagging annotation.
        """
        pos_prediction = self._model(document.text)
        document.pos_tags = [
            [(token.text, token.tag_) for token in sent if token.tag_ in self._considered_tags] for sent in pos_prediction.sents
        ]
        # keep pos tags only POS tags required for keyphrase extraction
        return self._extract_keyphrase_candidates(document)
