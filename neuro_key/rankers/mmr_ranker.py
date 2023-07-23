# Implementation borrowed from https://github.com/MaartenGr/KeyBERT/blob/master/keybert/_mmr.py
from operator import itemgetter
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass
from sentence_transformers import util
from torch import FloatTensor

from neuro_key.data.document import Document
from neuro_key.rankers import BaseRanker
from neuro_key.rankers.base_ranker import BaseRankerConfig


@dataclass
class MMRRankerConfig(BaseRankerConfig):
    """Configuration for the MMR ranker.

    Args:
        top_n: The number of keywords/keyhprases to return
        diversity: How diverse the select keywords/keyphrases are. Values between 0 and 1 with 0 being not
        diverse at all and 1 being most diverse.
    """

    diversity: float = 0.5
    top_n: int = 10


class MMRRanker(BaseRanker):
    def __init__(self, config: MMRRankerConfig):
        super().__init__(config)
        self._diversity = config.diversity
        self._top_n = config.top_n

    def _rank_keyphrase_candidates(
        self,
        document: Document,
        **kwargs,
    ) -> Document:
        """Calculate Maximal Marginal Relevance (MMR) between candidate keywords and the document.


        MMR considers the similarity of keywords/keyphrases with the document, along with the similarity of already selected
        keywords and keyphrases. This results in a selection of keywords that maximize their within diversity with respect
        to the document.

        Arguments:
            document: The document to extract keywords/keyphrases from.

        Returns:
            The document containing the ranked keywords/keyphrases.
        """
        # Extract similarity within words, and between words and the document
        document_embedding = FloatTensor(document.embedding)
        candidate_embeddings = FloatTensor(document.keyphrase_candidates_embeddings)
        word_doc_similarity = util.cos_sim(candidate_embeddings, document_embedding).cpu().numpy()
        word_similarity = util.cos_sim(candidate_embeddings, candidate_embeddings).cpu().numpy()
        # Initialize candidates and already choose best keyword/keyphrases
        keywords_idx = [np.argmax(word_doc_similarity)]
        keywords = getattr(document, "keyphrase_candidates", [])
        candidates_idx = [i for i in range(len(getattr(document, "keyphrase_candidates", []))) if i != keywords_idx[0]]

        for _ in range(min(self._top_n - 1, len(keywords) - 1)):
            # Extract similarities within candidates and between candidates and selected keywords/phrases
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

            # Calculate MMR
            mmr = (1 - self._diversity) * candidate_similarities - self._diversity * target_similarities.reshape(-1, 1)
            mmr_idx = candidates_idx[np.argmax(mmr)]

            # Update keywords & candidates
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        # Extract and sort keywords in descending similarity
        keywords = [(keywords[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4)) for idx in keywords_idx]
        keywords = sorted(keywords, key=itemgetter(1), reverse=True)

        document.keyphrase_candidates = keywords
        return document

    def _process(self, document: Document, **kwargs) -> Document:
        """Rank a single document.

        Args:
            document: A single document to be ranked.

        Returns:
            A single document with ranked keywords/keyphrases.
        """
        return self._rank_keyphrase_candidates(document, **kwargs)

    def _process_batch(self, documents: List[Document], **kwargs) -> List[Document]:
        """Rank a batch of documents.

        Args:
            documents: A batch of documents to be ranked.

        Returns:
            A batch of documents with ranked keywords/keyphrases.
        """
        return [self._rank_keyphrase_candidates(document=document, **kwargs) for document in documents]
