from nltk.parse import CoreNLPParser
from pydantic.dataclasses import dataclass

from neuro_key.data import Document
from neuro_key.pos_taggers import BasePOSTagger, BasePOSTaggerConfig


@dataclass
class CoreNLPPOSTaggerConfig(BasePOSTaggerConfig):
    """Config for CoreNLPPOSTagger.

    Args:
        host: The host of the CoreNLP server.
        port: The port of the CoreNLP server.
        pos_properties: The properties to use for the CoreNLP server.
    """

    host: str = "http://localhost"
    port: int = 9000
    # pos_properties: Dict[str, str] = {"annotators": "tokenize,ssplit,pos"}


class CoreNLPPOSTagger(BasePOSTagger):
    """CoreNLPPOSTagger for tagging POS tags.

    Args:
        config: The config for the CoreNLPPOSTagger.
    """

    def __init__(self, config: CoreNLPPOSTaggerConfig):
        super().__init__(config)
        self._host = config.host
        self._port = config.port
        self._parser = CoreNLPParser(url=f"{self._host}:{self._port}")

    def _get_pos_tags(self, document: Document) -> Document:
        """Get POS tags for a document.

        Args:
            document: A single document.

        Returns:
            A single document with POS tags.
        """
        text: str = getattr(document, self._content_field)
        tagged_data = self._parser.api_call(text)
        if not document.pos_tags:
            document.pos_tags = []
        for tagged_sentence in tagged_data["sentences"]:
            document.pos_tags.append([(token["word"], token["pos"]) for token in tagged_sentence["tokens"]])

        # keep pos tags only POS tags required for keyphrase extraction
        return self._extract_keyphrase_candidates(document)
